import os
import csv
from datetime import datetime
from argparse import Namespace

import cv2 as cv
import numpy as np
import torch
import torch.optim as optim
import wandb

# Import the INP model, loss function, and the firefighting environment.
from informed_meta_learning.models.rain import RAIN
from informed_meta_learning.models.loss import ELBOLoss
from gym_forest_fire.envs.energy_env import EnergyGridForestFireEnv


class FirefightingRAINTrainer:
    def __init__(self, config):
        """
        Initializes the trainer.
        
        Args:
            config: A Namespace object that contains parameters such as device, lr, num_tasks,
                    rollout_horizon, k_context, model architecture parameters, knowledge integration parameters,
                    and checkpoint_interval.
        """
        self.config = config
        self.device = config.device

        # Set a timestamp including date and time to group tasks.
        self.train_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize Weights & Biases.
        wandb.init(project="firefighting_rain_training", config=self.config)

        # Initialize the INP model (internally uses the knowledge encoder, etc.)
        self.model = RAIN(config).to(self.device)
        self.loss_func = ELBOLoss(beta=config.beta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        # Initialize best loss as infinity.
        self.best_loss = float('inf')

    def sample_task_env(self):
        """
        Creates a new firefighting environment with fixed random parameters
        and constructs a descriptive knowledge string.
        
        Returns:
            env: The firefighting environment instance.
            knowledge_str: A descriptive string about the environment.
        """
        env_kwargs = {
            "num_hospitals": 0,
            "num_power_plants": 1,
            "num_apartments": 2,
            "seed": 1
        }
        # Store env_kwargs for naming the folder later.
        self.current_env_kwargs = env_kwargs

        env = EnergyGridForestFireEnv(**env_kwargs)
        knowledge_str = (
            f"This environment has {env_kwargs['num_hospitals']} hospital(s), "
            f"{env_kwargs['num_power_plants']} power plant(s), and "
            f"{env_kwargs['num_apartments']} apartment(s)."
        )
        return env, knowledge_str

    def _to_tensor(self, data, unsqueeze_dims=0, flatten=False):
        """
        Helper to convert data to a torch tensor and move to the trainer's device.
        
        Args:
            data: The input data (numpy array or similar).
            unsqueeze_dims: Number of times to unsqueeze at the beginning.
            flatten: Whether to flatten the data.
        
        Returns:
            A torch tensor.
        """
        tensor = torch.tensor(np.array(data), dtype=torch.float32)
        if flatten:
            tensor = tensor.flatten()
        for _ in range(unsqueeze_dims):
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def select_action(self, context_states, context_expert_actions, current_state, knowledge):
        """
        Given the current context and state, use the INP to predict an action.
        If no context is available, returns a random action.
        
        Args:
            context_states: List of past states.
            context_expert_actions: List of past expert actions.
            current_state: The current state.
            knowledge: A descriptive string of the current task/environment.
        
        Returns:
            predicted_action: The action predicted by the model.
        """
        if len(context_states) == 0:
            return np.random.uniform(-1, 1, size=(self.config.output_dim,))
        else:
            x_context = self._to_tensor(context_states, unsqueeze_dims=1)
            y_context = self._to_tensor(context_expert_actions, unsqueeze_dims=1)
            if self.config.x_encoder == "mlp":
                current_state = self._to_tensor(current_state, unsqueeze_dims=2, flatten=True)
            else:
                current_state = self._to_tensor(current_state)
            with torch.no_grad():
                p_yCc, _, _, _ = self.model(x_context, y_context, current_state, None)
                predicted_action = p_yCc.sample().squeeze()
            return predicted_action

    def run_rollout(self, env, knowledge, save_images=False, task_idx=0):
        """
        Executes a full rollout (episode) in the environment with training updates.
        Optionally saves rendered images of the rollout.
        
        Args:
            env: The environment instance.
            knowledge: The knowledge string for the current task.
            save_images: Boolean flag to determine if images should be saved.
            task_idx: Current task index (for naming saved images).
        
        Returns:
            losses: List of loss values at each timestep.
            save_dir: Directory where images are saved (if applicable).
            total_reward: Total reward accumulated during the rollout.
        """
        losses = []
        context_states = []
        context_expert_actions = []
        total_reward = 0.0

        obs, info = env.reset()
        done = False
        t = 0

        print(f"Starting new task: {knowledge}")

        save_dir = None
        if save_images:
            env.render_mode_ = "rgb_array"
            current_datetime = self.train_timestamp
            env_params = self.current_env_kwargs
            prefix = "rain" if self.config.rag else "inp"
            base_dir = os.path.join("images", current_datetime)
            os.makedirs(base_dir, exist_ok=True)
            save_dir = os.path.join(
                base_dir,
                f"{prefix}_seed{self.config.seed}_"
                f"hosp{env_params['num_hospitals']}_"
                f"power{env_params['num_power_plants']}_"
                f"apart{env_params['num_apartments']}_"
                f"task_{task_idx}"
            )
            os.makedirs(save_dir, exist_ok=True)

        # Initialize values for loss and reward.
        prev_reward = 0.0
        loss, kl, nll = None, None, None

        while not done and t < self.config.rollout_horizon:
            # Process state based on encoder type.
            state = obs.flatten() if self.config.x_encoder == "mlp" else obs

            # Retrieve expert action from info; if missing, compute it.
            expert_action = info.get("best_action", None)
            if expert_action is None:
                expert_action, _ = env.compute_best_action()

            if t >= self.config.k_context:
                # Prepare context and target tensors.
                x_context = self._to_tensor(context_states[-self.config.k_context:], unsqueeze_dims=1)
                y_context = self._to_tensor(context_expert_actions[-self.config.k_context:], unsqueeze_dims=1)
                if self.config.x_encoder == "cnn":
                    x_target = self._to_tensor(state)
                elif self.config.x_encoder == "mlp":
                    x_target = self._to_tensor(state, unsqueeze_dims=2, flatten=True)
                y_target = self._to_tensor(expert_action, unsqueeze_dims=2)

                # Forward pass and loss computation.
                p_yCc, z_samples, q_zCc, q_zCct = self.model(x_context, y_context, x_target, y_target)
                loss, kl, nll = self.loss_func((p_yCc, z_samples, q_zCc, q_zCct), y_target)
                self.optimizer.zero_grad()
                # loss -= reward
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            # Select action using model or random if no context.
            action = self.select_action(context_states, context_expert_actions, state, knowledge)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if t >= self.config.k_context:
                best_reward = info.get("best_reward", None)
                print(
                    f"Step {t:<5}  "
                    f"Loss: {loss:<8.3f}  "
                    f"KL: {kl:<8.5f}  "
                    f"NLL: {nll:<8.3f}  "
                    f"Reward: {reward:<8.3f}  "
                    f"Best Reward: {best_reward:<8.3f}  "
                    f"Top-k Values: {np.round(self.model._topk_values.squeeze().detach().numpy(), 2)}"
                    f"\t Top-k Titles: {self.model._topk_titles}"
                )
                wandb.log({
                    "step_loss": loss.item(),
                    "step_kl": kl.mean().item(),
                    "step_nll": nll.mean().item(),
                    "reward": reward,
                    "best_reward": best_reward,
                    "step": t,
                    "task_idx": task_idx
                })

            if save_images:
                img = env.render()
                if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                    img_path = os.path.join(save_dir, f"frame_{t}.png")
                    cv.imwrite(img_path, img)

            # Save current state and expert action to context.
            context_states.append(self._to_tensor(obs))
            context_expert_actions.append(expert_action)

            prev_reward = reward
            t += 1

        # Final loss computation after the rollout.
        if t >= self.config.k_context:
            if self.config.x_encoder == "cnn":
                x_target = self._to_tensor(state)
            elif self.config.x_encoder == "mlp":
                x_target = self._to_tensor(obs.flatten(), unsqueeze_dims=2)
            y_target = self._to_tensor(expert_action, unsqueeze_dims=2)
            x_context = self._to_tensor(context_states[-self.config.k_context:], unsqueeze_dims=1)
            y_context = self._to_tensor(context_expert_actions[-self.config.k_context:], unsqueeze_dims=1)
            final_out = self.model(x_context, y_context, x_target, y_target)
            final_loss, final_kl, final_nll = self.loss_func(final_out, y_target)
            # Adjust final loss with the last reward.
            final_loss = final_loss - reward
            losses.append(final_loss.item())

            print(
                f"Step {t:<5}  "
                f"Loss: {final_loss:<8.3f}  "
                f"KL: {final_kl.mean():<8.5f}  "
                f"NLL: {final_nll.mean():<8.3f}  "
                f"Reward: {reward:<8.3f}  "
                f"Best Reward: {best_reward:<8.3f}  "
                f"Top-k Values: {np.round(self.model._topk_values.squeeze().detach().numpy(), 2)}"
                f"\t Top-k Titles: {self.model._topk_titles}"
            )
            wandb.log({
                "step_loss": final_loss.item(),
                "step_kl": final_kl.mean().item(),
                "step_nll": final_nll.mean().item(),
                "reward": reward,
                "best_reward": best_reward,
                "task_idx": task_idx,
                "step": t
            })

        print(f"Task completed in {t} timesteps.\n")
        return losses, save_dir, total_reward

    def save_top_documents(self, save_dir):
        """
        Saves the top-k retrieved documents (titles, similarity scores, and text) to a CSV file
        in the provided directory, and logs the table to wandb.
        
        Args:
            save_dir: Directory to save the CSV file.
        """
        file_path = os.path.join(save_dir, "top_documents.csv")
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Title", "Similarity Score", "Document"])
            for title, score in zip(self.model._topk_titles, self.model._topk_values.squeeze().tolist()):
                idx = self.model.titles.index(title)
                doc_text = self.model.documents[idx]
                writer.writerow([title, f"{score:.3f}", doc_text])
        print(f"Top {self.config.top_k} documents saved to {file_path}")
        wandb.log({
            "top_documents_csv": wandb.Table(
                data=[
                    [title, score, self.model.documents[self.model.titles.index(title)]]
                    for title, score in zip(
                        self.model._topk_titles,
                        self.model._topk_values.squeeze().tolist()
                    )
                ],
                columns=["Title", "Similarity Score", "Document"]
            )
        })

    def save_checkpoint(self, iteration, save_dir):
        """
        Saves a model checkpoint containing the model state, optimizer state, and iteration number.
        
        Args:
            iteration: Current iteration number.
            save_dir: Directory where the checkpoint will be saved.
        """
        checkpoint_path = os.path.join(save_dir, f"checkpoint_{iteration}.pth")
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        print(f"Checkpoint saved at iteration {iteration} to {checkpoint_path}")
        wandb.log({"checkpoint_saved": iteration, "checkpoint_path": checkpoint_path})

    def save_best_model(self, iteration, loss, save_dir):
        """
        Saves the best model checkpoint (with the lowest loss) to a file named 'best_model.pth'.
        
        Args:
            iteration: Current iteration number.
            loss: Loss value corresponding to the best model.
            save_dir: Directory where the checkpoint will be saved. If None, uses a global 'checkpoints' folder.
        """
        if save_dir is None:
            global_dir = "checkpoints"
            os.makedirs(global_dir, exist_ok=True)
            best_model_path = os.path.join(global_dir, "best_model.pth")
        else:
            best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save({
            'iteration': iteration,
            'loss': loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, best_model_path)
        print(f"New best model (loss {loss:.3f}) saved at iteration {iteration} to {best_model_path}")
        wandb.log({"best_model_saved": iteration, "best_loss": loss, "best_model_path": best_model_path})

    def train_one_task(self, task_idx, save_images=False):
        """
        Samples a new environment task, runs a rollout with training updates, and logs metrics.
        Also appends and then removes the task's knowledge from the model's document store.
        
        Args:
            task_idx: Index of the current task.
            save_images: Boolean flag for saving rendered images.
        
        Returns:
            A tuple containing the average loss over the rollout, the save directory (if applicable), and the total reward.
        """
        env, knowledge_str = self.sample_task_env()
        knowledge = knowledge_str

        # Append current task's knowledge to the model's document store.
        self.model.titles.append("Fire Task Knowledge")
        self.model.documents.append(knowledge)
        document_embedding = self.model.prepare_document(knowledge)
        self.model.document_embeddings = torch.vstack([self.model.document_embeddings, document_embedding])

        losses, save_dir, total_reward = self.run_rollout(env, knowledge, save_images=save_images, task_idx=task_idx)

        if save_dir is not None:
            # self.save_top_documents(save_dir)
            if task_idx % self.config.checkpoint_interval == 0:
                self.save_checkpoint(task_idx, save_dir)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        wandb.log({
            "task_avg_loss": avg_loss,
            "task_total_reward": total_reward,
            "task_idx": task_idx
        })

        # Remove the task's knowledge from the model's document store.
        self.model.documents.pop()
        self.model.titles.pop()
        self.model.document_embeddings = self.model.document_embeddings[:-1, :]

        return avg_loss, save_dir, total_reward

    def train_loop(self, num_tasks=None):
        """
        Main training loop over tasks.
        
        Args:
            num_tasks: Number of tasks to train on. If None, uses self.config.num_tasks.
        """
        num_tasks = num_tasks if num_tasks is not None else self.config.num_tasks
        task_losses = []

        for task_idx in range(num_tasks):
            save_images = self.config.save_images and (task_idx % self.config.save_interval == 0)
            loss_val, current_save_dir, total_reward = self.train_one_task(task_idx, save_images=save_images)
            if loss_val is not None:
                task_losses.append(loss_val)
            print(f"Task {task_idx}/{num_tasks}, loss: {loss_val:.3f}")
            wandb.log({"global_task_loss": loss_val, "global_task_idx": task_idx})
            # Save best model if current loss is lower than the best so far.
            if current_save_dir is not None and loss_val < self.best_loss:
                self.best_loss = loss_val
                self.save_best_model(task_idx, loss_val, current_save_dir)
        print("Training complete!")
        wandb.log({"final_best_loss": self.best_loss})


if __name__ == "__main__":
    # Create a dummy configuration using Namespace.
    config = Namespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=1e-4,
        num_tasks=2000,
        rollout_horizon=50,
        k_context=20,
        seed=0,
        beta=1.0,
        input_dim=4096 * 3,  # Flattened 64x64x3 grid
        output_dim=2,        # Action dimension
        hidden_dim=128,
        xy_encoder_num_hidden=2,
        xy_encoder_hidden_dim=384,  # e.g., 3 * hidden_dim
        data_agg_func="mean",
        latent_encoder_num_hidden=2,
        decoder_hidden_dim=128,
        decoder_num_hidden=3,
        decoder_activation="gelu",
        x_encoder="cnn",
        x_transf_dim=128,
        x_encoder_num_hidden=1,
        use_knowledge=True,
        text_encoder="roberta",
        freeze_llm=False,
        tune_llm_layer_norms=False,
        knowledge_dropout=0.3,
        knowledge_dim=128,
        knowledge_merge="sum",
        knowledge_extractor_num_hidden=2,
        knowledge_extractor_hidden_dim=128,
        knowledge_input_dim=128,
        train_num_z_samples=1,
        test_num_z_samples=16,
        batch_size=1,
        save_images=True,
        save_interval=10,
        checkpoint_interval=50,
        rag=True,
        documents_path="informed_meta_learning/data/documents/data.json",
        query_encoder="inp",
        similarity_metric="mips",
        top_k=3,
        verbose=True
    )
    trainer = FirefightingRAINTrainer(config)
    trainer.train_loop()
