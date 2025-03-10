import torch
import torch.optim as optim
import numpy as np
from argparse import Namespace
import cv2 as cv
import os

# Import the INP model, loss function, and the firefighting environment.
from informed_meta_learning.models.inp import INP
from informed_meta_learning.models.loss import ELBOLoss
from gym_forest_fire.envs.energy_env import EnergyGridForestFireEnv  

class FirefightingINPTrainer:
    def __init__(self, config):
        """
        config should contain:
          - device, lr, num_tasks, rollout_horizon, k_context, etc.
          - Model architecture parameters (input_dim, output_dim, etc.)
          - Knowledge integration parameters
        """
        self.config = config
        self.device = config.device

        # Initialize the INP model (which internally uses the knowledge encoder, etc.)
        self.model = INP(config).to(self.device)
        self.loss_func = ELBOLoss(beta=config.beta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    def sample_task_env(self):
        """
        Instantiate a new firefighting environment with random parameters.
        Also constructs a knowledge string describing the environment.
        """
        env_kwargs = {
            "num_hospitals": 1,
            "num_power_plants": 1,
            "num_apartments": np.random.randint(1, 4),
            "seed": 1
        }
        env = EnergyGridForestFireEnv(**env_kwargs)
        knowledge_str = (
            f"This environment has {env_kwargs['num_hospitals']} hospital(s), "
            f"{env_kwargs['num_power_plants']} power plant(s), and "
            f"{env_kwargs['num_apartments']} apartment(s)."
        )
        return env, knowledge_str

    def select_action(self, context_states, context_expert_actions, current_state, knowledge):
        """
        Given the current context and the current state, use the INP to predict an action.
        If no context is available, return a random action.
        """
        # TODO: maybe implement different strats
        if len(context_states) == 0:
            return np.random.uniform(-1, 1, size=(self.config.output_dim,))
        else:
            x_context = torch.tensor(np.array(context_states), dtype=torch.float32).unsqueeze(0).to(self.device)
            y_context = torch.tensor(np.array(context_expert_actions), dtype=torch.float32).unsqueeze(0).to(self.device)
            current_state_flat = torch.tensor(current_state, dtype=torch.float32).flatten().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p_yCc, _, _, _ = self.model(x_context, y_context, current_state_flat, None, knowledge)
                # predicted_action = p_yCc.mean.squeeze(0).squeeze(0).cpu().numpy()
                predicted_action = p_yCc.sample().squeeze()
            return predicted_action

    def run_rollout(self, env, knowledge, save_images=False, task_idx=0):
        """
        Roll out one full task (episode) in the environment.
        Now, after we have at least k_context steps, we compute a training update
        at every timestep using the most recent k_context observations as context and
        the current step as the target.
        Optionally saves rendered images of the rollout.
        Returns:
          - losses: list of per-timestep loss values computed during the rollout.
        """
        losses = []
        context_states = []
        context_expert_actions = []
        
        obs, info = env.reset()
        done = False
        t = 0
        
        print(f"Starting new task: {knowledge}")
        
        # Define directory for saving images if enabled
        if save_images:
            env.render_mode_ = "rgb_array"
            save_dir = f"images/task_{task_idx}"
            os.makedirs(save_dir, exist_ok=True)
        
        while not done and t < self.config.rollout_horizon:
            state_flat = np.array(obs).flatten()
            
            # Retrieve the expert (best) action for the current state.
            expert_action = info.get("best_action", None)
            if expert_action is None:
                expert_action, _ = env.compute_best_action()
            
            # If we have enough context, perform a training update on this timestep.
            if t >= self.config.k_context:
                x_context = torch.tensor(np.array(context_states[-self.config.k_context:]), dtype=torch.float32).unsqueeze(0).to(self.device)
                y_context = torch.tensor(np.array(context_expert_actions[-self.config.k_context:]), dtype=torch.float32).unsqueeze(0).to(self.device)
                x_target = torch.tensor(np.array(state_flat), dtype=torch.float32).flatten().unsqueeze(0).unsqueeze(0).to(self.device)
                y_target = torch.tensor(np.array(expert_action), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                
                p_yCc, z_samples, q_zCc, q_zCct = self.model(x_context, y_context, x_target, y_target, knowledge)
                loss, kl, nll = self.loss_func((p_yCc, z_samples, q_zCc, q_zCct), y_target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                # print(f"Step {t}: Loss: {loss:.3f}, KL: {kl:.3f}, NLL: {nll:.3f}, Reward: {reward:.3f}, Best Reward: {best_reward:.3f}")
            
            # Select an action using the current (possibly smaller) context.
            action = self.select_action(context_states, context_expert_actions, state_flat, knowledge)
            
            # Take a step in the environment.
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if t >= self.config.k_context:
                best_reward = info.get("best_reward", None)
                print(f"Step {t}: Loss: {loss:.3f}, KL: {kl:.3f}, NLL: {nll:.3f}, Reward: {reward:.3f}, Best Reward: {best_reward:.3f}")
            
            # breakpoint()
            if save_images:
                img = env.render()
                if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                    img_path = os.path.join(save_dir, f"frame_{t}.png")
                    cv.imwrite(img_path, img)
            
            # Append the current state and expert action to the context.
            context_states.append(state_flat)
            context_expert_actions.append(expert_action)
            
            t += 1
        
        print(f"Task completed in {t} timesteps.\n")
        return losses


    def train_one_task(self, save_images=False):
        """
        Samples a new environment task, collects a full rollout (with per-timestep training updates),
        and returns the average loss over the rollout (if any updates were performed).
        """
        env, knowledge_str = self.sample_task_env()
        knowledge = knowledge_str
        
        losses = self.run_rollout(env, knowledge, save_images=save_images)
        
        if len(losses) > 0:
            avg_loss = sum(losses) / len(losses)
        else:
            avg_loss = 0.0
        
        return avg_loss


    def train_loop(self, num_tasks=None):
        """
        Main training loop: iterate over tasks and update the model.
        """
        num_tasks = num_tasks if num_tasks is not None else self.config.num_tasks
        task_losses = []
        for task_idx in range(num_tasks):
            save_images = self.config.save_images and (task_idx % self.config.save_interval == 0)
            loss_val = self.train_one_task(save_images=save_images)
            if loss_val is not None:
                task_losses.append(loss_val)
            print(f"Task {task_idx}/{num_tasks}, loss: {loss_val:.3f}")
        print("Training complete!")

if __name__ == "__main__":
    # Create a dummy config using Namespace (similar to inp.py)
    config = Namespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=1e-3,
        num_tasks=2000,
        rollout_horizon=105,
        k_context=5,
        seed=0,
        beta=1.0,
        input_dim=4096,       # Flattened 64x64 grid
        output_dim=2,         # Action dimension
        hidden_dim=128,
        xy_encoder_num_hidden=2,
        xy_encoder_hidden_dim=384,  # e.g., 3 * hidden_dim
        data_agg_func="sum",
        x_encoder="mlp",
        latent_encoder_num_hidden=1,
        decoder_hidden_dim=128,
        decoder_num_hidden=3,
        decoder_activation="gelu",
        x_transf_dim=128,
        x_encoder_num_hidden=1,
        use_knowledge=3,
        text_encoder="roberta",
        freeze_llm=True,
        tune_llm_layer_norms=False,
        knowledge_dropout=0.3,
        knowledge_dim=128,
        knowledge_merge="sum",
        knowledge_extractor_num_hidden=2,
        knowledge_extractor_hidden_dim=128,
        knowledge_input_dim=128,
        train_num_z_samples=1,   # Newly added attribute
        test_num_z_samples=16,    # Newly added attribute
        batch_size=1,
        save_images=True,
        save_interval=10,
    )
    trainer = FirefightingINPTrainer(config)
    trainer.train_loop()
