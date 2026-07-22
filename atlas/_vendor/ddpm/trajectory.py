import torch
import torch.nn.functional as F
from .BiFlowNet import GaussianDiffusion


class Trajectory(GaussianDiffusion):

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
    def compute_deformation_field(
        self, denoise_fn, x0, y, res, timesteps
    ):
        """
        Deterministic atlas transport:
        - no noise
        - deformation is geometry-consistent
        """
        xt = x0.clone()
        accumulated_flow = torch.zeros_like(xt)
        flows = []

        for i, t in enumerate(timesteps):
            b = xt.shape[0]
            batched_t = torch.full((b,), t, device=xt.device, dtype=torch.long)

            # xt = F.interpolate(
            #     xt, size=original_size, mode='trilinear', align_corners=False)

            
            # if t > 900:
            #     t_prev_batch = torch.full((b,), timesteps[i + 1], device=xt.device, dtype=torch.long)
            #     x_prev = self.p_sample_ddim(
            #         denoise_fn=denoise_fn,
            #         x=xt,
            #         y=y,
            #         res=res,
            #         t=batched_t,
            #         t_prev=t_prev_batch,
            #         clip_denoised=True
            #     )
            # else:

            model_mean, _, model_log_variance = self.p_mean_variance(
                denoise_fn=denoise_fn,
                x=xt,
                y=y,
                res=res,
                t=batched_t,
                clip_denoised=True
            )

            x_prev = model_mean                  # deterministic x_{t-1}

            drift = x_prev - xt                  # deformation increment

            accumulated_flow += drift
            flows.append(drift)

            xt = x_prev

        atlas = xt
        deformation = accumulated_flow

        return atlas, deformation, flows

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

        for i in range(len(timesteps) - 1):
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
    def atlas_project_x0(self, x, t, clip_denoised=True):
        """
        Returns DIA-style one-step atlas projection: x_hat0 = predicted x_start.
        """
        B = x.shape[0]
        batched_times = torch.full((B,), t, device=x.device, dtype=torch.long)
        _, _, _, x_start = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised
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