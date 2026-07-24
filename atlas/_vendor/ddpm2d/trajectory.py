import torch
from .model import GaussianDiffusion


class Trajectory(GaussianDiffusion):
    
    @torch.no_grad()
    def atlas_transport_from_score(
        self,
        x0,
        timesteps,
        dt=1
    ):
        x = torch.zeros_like(x0)
        xs = [x0]

        for idx, t in enumerate(timesteps[1:]):
            t_prev = timesteps[idx + 1]

            tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            preds = self.model_predictions(x, tt, x_self_cond=None)
            eps = preds.pred_noise
            x0_hat = preds.pred_x_start

            a_prev = self.alphas_cumprod[t_prev]

            # deterministic DDIM step
            x = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps

            xs.append(x)

        return xs[-1]  # sequential trajectory x_t -> ... -> x_{t_star}

    @torch.no_grad()
    def generate_atlas_via_bottleneck(self, x_input, t_max=999):
        """
        Generates the Population Atlas by passing a subject through the
        diffusion 'information bottleneck'.
        
        1. Encodes x_input to pure noise (removing individual features).
        2. Decodes deterministically (restoring only population features).
        
        Args:
            x_input: A batch of sample images (B, C, H, W). 
                    (Ideally use 4-8 diverse subjects to verify convergence).
            t_max:   The max diffusion step (usually 999).
        """
        
        # -----------------------------------------------------
        # Step 1: Forward Diffusion (The Information Bottleneck)
        # -----------------------------------------------------
        # We push the images to the noise limit. 
        # At t=999, x_t contains almost zero information about the specific subject.
        
        # noise = torch.randn_like(x_input)
        
        # Extract alpha_cumprod for the max step
        # (Assuming standard DDPM schedule access)
        alpha_bar = self.alphas_cumprod[t_max] 
        
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
        # x_t = torch.sqrt(alpha_bar) * x_input + torch.sqrt(1 - alpha_bar) * noise

        # -----------------------------------------------------
        # Step 2: Deterministic Reverse Transport (The Atlas Discovery)
        # -----------------------------------------------------
        # We traverse the probability flow ODE from Noise -> Data.
        # Crucially: NO new noise is injected.

        current_img = x_input
        x_start = None
        # Iterate backwards: 999 -> 0
        # Using 'tqdm' is recommended for progress tracking
        times = torch.linspace(-1, t_max - 1, steps=self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        for i, (t, t_next) in enumerate(time_pairs):
            # Create batch of current timestep
            t_batch = torch.full((x_input.shape[0],), t, device=x_input.device, dtype=torch.long)
            
            # 1. Predict the Model Output (Noise or Mean)
            # We rely on the model to guide us to the 'center' of the manifold
            # model_output = self.model(current_img, t_batch)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(current_img, t_batch, self_cond)
            x_start = preds.pred_x_start
            pred_noise = preds.pred_noise

            # 3. Determine the 'Drift' (Direction to move)
            # To be deterministic, we point towards the posterior mean
            # This is effectively DDIM with eta=0.0

            if i == len(time_pairs) - 1:
                current_img = x_start
                continue

            # Get alpha for next step (t-1)
            if t > 0:
                alpha_bar_prev = self.alphas_cumprod[t-1]
            else:
                alpha_bar_prev = torch.tensor(1.0).to(x_input.device)

            # DDIM Update Direction (Equation 12 in DDIM paper)
            # Direction pointing to x_t-1
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise

            # New State
            x_prev = torch.sqrt(alpha_bar_prev) * x_start + dir_xt

            current_img = x_prev

        # The result is the Atlas
        return current_img

    @torch.no_grad()
    def compute_deformation_field_old(self, x0, timesteps):
        """
        x0:    the noisy input (or noisy version x_T)
        timesteps: e.g. reversed(range(T))
        """

        xt = x0.clone()
        accumulated_flow = torch.zeros_like(xt)  # deformation accumulator

        flows = []  # store flow at each step for visualization

        for t in timesteps:
            # get next prediction
            x_next, x_start = self.p_sample(xt, t)

            # --- deformation drift ---
            # deterministic direction (the mean update)
            drift = (x_next - xt)

            accumulated_flow += drift
            flows.append(drift)

            xt = x_next

        # final flow: sum of all incremental drifts
        deformation = accumulated_flow
        return xt, deformation, flows

    @torch.no_grad()
    def compute_atlas(self, x0, timesteps, eta=0.1):
        """
        Deterministic atlas transport:
        - no noise
        - deformation is geometry-consistent
        """

        xt = x0.clone()

        for t in timesteps:
            b = xt.shape[0]
            batched_t = torch.full((b,), t, device=xt.device, dtype=torch.long)

            model_mean, _, model_log_variance, x_start = self.p_mean_variance(
                x=xt,
                t=batched_t,
                clip_denoised=True
            )
            # if t > 980:
            # noise = torch.randn_like(model_mean) if t > 0 else 0.
            # x_prev = model_mean + (0.5 * model_log_variance).exp() * noise * eta
            # else:
            x_prev = model_mean                  # deterministic x_{t-1}
            # drift = x_prev - xt                  # deformation increment

            xt = x_prev

        atlas = x_prev

        return atlas

    @torch.no_grad()
    def compute_atlas_flow_ode(self, xT, timesteps):
        """
        Probability-flow atlas transport
        - deterministic
        - defines a true continuous drift field
        """

        xt = xT.clone()
        accumulated_flow = torch.zeros_like(xt)
        flows = []

        for i in reversed(range(len(timesteps) - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            b = xt.shape[0]
            batched_t = torch.full((b,), t, device=xt.device, dtype=torch.long)

            # predict x0
            _, _, _, x0_hat = self.p_mean_variance(
                x=xt,
                t=batched_t,
                clip_denoised=True
            )

            alpha_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            # probability flow drift
            drift = -0.5 * beta_t * (
                xt - x0_hat / torch.sqrt(alpha_t)
            )

            dt = t_next - t
            dx = drift * dt

            xt = xt + dx
            flows.append(dx)
            accumulated_flow += dx

        atlas = xt
        deformation = accumulated_flow

        return atlas, deformation, flows

    @torch.no_grad()
    def compute_atlas_flow_ddim(self, x_t, timesteps, *, eta=0.0, step_scale=1.0):
        """
        Deterministic DDIM-style march while *still* tracking an "atlas deformation".

        This tends to be more stable than xt <- x0_hat directly, because it respects
        the diffusion geometry.

        Requires your model to expose:
          - self.alphas_cumprod (Tensor length T)
          - self.predict_eps_from_xstart(x_t, t, x0_hat)  (or implement below)

        Returns:
          atlas:        final x at last timestep
          deformation:  sum(dx)
          flows:        list(dx)
        """

        xt = x_t.clone()
        accumulated_flow = torch.zeros_like(xt)
        flows = []

        alphas = self.alphas_cumprod  # [T]

        def predict_eps_from_xstart(x_t, t, x0):
            # epsilon = (x_t - sqrt(alpha_t) * x0) / sqrt(1 - alpha_t)
            a = alphas[t].to(x_t.device)
            return (x_t - torch.sqrt(a) * x0) / torch.sqrt(1.0 - a)

        for i in range(len(timesteps) - 1):
            t = int(timesteps[i])
            t_next = int(timesteps[i + 1])

            b = xt.shape[0]
            batched_t = torch.full((b,), t, device=xt.device, dtype=torch.long)

            # x0 prediction
            _, _, _, x0_hat = self.p_mean_variance(
                x=xt,
                t=batched_t,
                clip_denoised=True
            )

            eps_hat = predict_eps_from_xstart(xt, t, x0_hat)

            a_t = alphas[t].to(xt.device)
            a_next = alphas[t_next].to(xt.device)

            # DDIM deterministic update (eta=0)
            x_next = (
                torch.sqrt(a_next) * x0_hat
                + torch.sqrt(1.0 - a_next) * eps_hat
            )

            dx = step_scale * (x_next - xt)
            xt = xt + dx

            flows.append(dx)
            accumulated_flow += dx

        atlas = xt
        deformation = accumulated_flow
        return atlas, deformation, flows
    
    @torch.no_grad()
    def probability_flow_drift(self, x, t):
        """
        x: [B, C, H, W]
        t: integer timestep
        """

        # DDPM noise prediction
        eps = self.p_sample(x, t)[0]

        beta_t = self.betas[t]
        alpha_bar_t = self.alphas_cumprod[t]

        drift = - 0.5 * beta_t / torch.sqrt(1 - alpha_bar_t) * eps
        return drift
    
    @torch.no_grad()
    def integrate_probability_flow(
        self,
        x0,
        t_start,
        t_end=0,
        steps=200
    ):
        """
        Integrate dx/dt = v_theta(x, t)

        x0: initial image
        returns: atlas-projected image
        """

        x = x0.clone()
        ts = torch.linspace(t_start, t_end, steps, device=x.device)

        for i in range(len(ts) - 1):
            t = int(ts[i].item())
            dt = ts[i+1] - ts[i]

            v = self.probability_flow_drift(x, t)
            x = x + dt * v

        return x
    
    @torch.no_grad()
    def atlas_project_x0(self, x, t, x_self_cond=None, clip_denoised=True):
        """
        Returns DIA-style one-step atlas projection: x_hat0 = predicted x_start.
        """
        B = x.shape[0]
        batched_times = torch.full((B,), t, device=x.device, dtype=torch.long)
        _, _, _, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )
        return x_start
    
    @torch.no_grad()
    def integrate_projection_flow(
        self, x, t,
        steps=20, step_size=1.0,
        clip_denoised=True
    ):
        """
        ODE on artificial time tau:
            dx/dtau = x_hat0(x,t) - x
        """
        xt = x.clone()
        for _ in range(steps):
            x_hat0 = self.atlas_project_x0(xt, t, clip_denoised=clip_denoised)
            xt = xt + step_size * (x_hat0 - xt)
        return xt
    
    @torch.no_grad()
    def ddim_inversion(
        self,
        x0,
        timesteps,          # increasing order: [0, ..., T]
    ):
        """
        Deterministic DDIM inversion:
        maps x_0 -> x_T along the model's trajectory
        """

        xt = x0.clone()
        xs = [xt]

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            b = xt.shape[0]
            t_batch = torch.full((b,), t, device=xt.device, dtype=torch.long)

            # ---- model prediction ----
            preds = self.model_predictions(xt, t_batch)
            x0_hat = preds.pred_x_start

            # ---- recover epsilon ----
            alpha_t = self.alphas_cumprod[t]
            eps = (xt - alpha_t.sqrt() * x0_hat) / (1 - alpha_t).sqrt()

            # ---- deterministic forward step ----
            alpha_next = self.alphas_cumprod[t_next]
            xt = alpha_next.sqrt() * x0_hat + (1 - alpha_next).sqrt() * eps

            xs.append(xt)

        return xt, xs   # x_T and all intermediates

    @torch.no_grad()
    def probability_flow_atlas(
        model,
        shape,
        alphas,
        sigmas,
        timesteps,
        device="cuda"
    ):
        """
        Input-free atlas via probability-flow ODE.
        Deterministic, no noise, no data.

        Args:
            model: DDPM noise predictor eps_theta(x_t, t)
            shape: (B, C, H, W)
            alphas, sigmas: diffusion schedule tensors of length T
            timesteps: list like [T-1, ..., 0]
        """

        # Fixed isotropic initialization (symmetry anchor)
        x = torch.zeros(shape, device=device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            dt = t_next - t  # negative

            alpha_t = self.alphas_cumprod[t]
            sigma_t = self.sigmas[t]

            b = x.shape[0]
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            eps = model(x, t_batch)

            drift = 0.5 * (alpha_t.log().diff() * (x - sigma_t * eps))

            # Euler integration
            x = x + drift * dt

        return x

    @torch.no_grad()
    def score_null_atlas(
        model,
        shape,
        t_star,
        n_iters=200,
        lr=0.1,
        device="cuda"
    ):
        """
        Finds a stationary point of the score field.
        No data, no noise, no sampling.

        Args:
            t_star: fixed diffusion timestep
        """

        x = torch.zeros(shape, device=device)

        alpha_t = self.alphas[t_star]
        sigma_t = self.sigmas[t_star]

        for _ in range(n_iters):
            b = x.shape[0]
            t_batch = torch.full((b,), t_star, device=device, dtype=torch.long)

            eps = model(x, t_batch)

            # Tweedie denoised estimate
            x0_hat = (x - sigma_t * eps) / alpha_t.sqrt()

            update = x0_hat - x
            x = x + lr * update

            if update.abs().mean() < 1e-5:
                break

        return x