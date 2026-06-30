# vb, 10.03.2026

import os
import numpy as np
import matplotlib.pyplot as plt
from sampler_utils_RC3D import *


# ########################################## visualise strains ##########################################

# save_data_dir = os.path.join('C:\\', 'kuy_sampling_strain-stress')
# save_data_path = os.path.join(save_data_dir, 'output_eps_g.h5')
# plot_3D_data(save_data_path, filename = 'scatter_eps_g', n_every = int(1))

# # TODO?
# # visualisation of stress distributions across the height of the cross section?

# ########################################## visualise stresses ##########################################

# save_data_dir = os.path.join('C:\\', 'kuy_sampling_strain-stress')
# save_data_path = os.path.join(save_data_dir, 'output_sig_g.h5')
# plot_3D_data(save_data_path, filename = 'scatter_sig_g', n_every = int(1))


# Ergänzung kuy, 29.06.2026
########################################## read-in data ##########################################
save_data_dir = os.path.join('C:\\', 'kuy_sampling_strain-stress')

save_data_path = os.path.join(save_data_dir, 'output_eps_g.h5')
data_eps_g = read_h5_file(save_data_path,  filename = 'eps_g', n_every = int(1))

save_data_path = os.path.join(save_data_dir, 'output_sig_g.h5')
data_sig_g = read_h5_file(save_data_path,  filename = 'sig_g', n_every = int(1))

save_data_path = os.path.join(save_data_dir, 'output_e.h5')
data_e = read_h5_file(save_data_path,  filename = 'e', n_every = int(1))

save_data_path = os.path.join(save_data_dir, 'output_e_princ.h5')
data_e_princ = read_h5_file(save_data_path,  filename = 'e_princ', n_every = int(1))

# Set this to the first step you want to run.
# x means start directly from the regime-specific analysis block. (currently 10 always needs 9 to be run first, since regime_ids are generated there)
START_FROM_STEP = 10

plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(plot_dir, exist_ok=True)


########################################## analysis of strain distribution to regimes ##########################################

print("\n=== GLOBAL STRESS RANGES (full dataset) ===")

print("n_x  min/max:", np.min(data_sig_g[:,0]), np.max(data_sig_g[:,0]))
print("n_y  min/max:", np.min(data_sig_g[:,1]), np.max(data_sig_g[:,1]))
print("n_xy min/max:", np.min(data_sig_g[:,2]), np.max(data_sig_g[:,2]))

print("m_x  min/max:", np.min(data_sig_g[:,3]), np.max(data_sig_g[:,3]))
print("m_y  min/max:", np.min(data_sig_g[:,4]), np.max(data_sig_g[:,4]))
print("m_xy min/max:", np.min(data_sig_g[:,5]), np.max(data_sig_g[:,5]))


# --------------------------------------------------
# 1. Extract principal strains
# --------------------------------------------------

# data_e_princ shape: (N, 20, 3)
# assuming:
# e1 = largest principal strain
# e3 = smallest principal strain

e1 = data_e_princ[:, :, 0]
e3 = data_e_princ[:, :, 1]
th = data_e_princ[:, :, 2]

violations = np.sum(e1 < e3)

if violations > 0:
    print("Violations detected:", violations)
else:
    print("All good: e1 >= e3 everywhere")


# --------------------------------------------------
# 2. Classify sign combinations
# --------------------------------------------------

# we only need 3 states now (since e1 >= e3 eliminates -+):
# 1 → (e1 > 0 , e3 > 0)   → ++
# 2 → (e1 > 0 , e3 < 0)   → +-
# 3 → (e1 < 0 , e3 < 0)   → --

def classify_signs(e1, e3):
    signs = np.zeros(e1.shape, dtype=int)

    signs[(e1 > 0) & (e3 > 0)] = 1   # ++
    signs[(e1 > 0) & (e3 < 0)] = 2   # +-
    signs[(e1 < 0) & (e3 < 0)] = 3   # --
    signs[(e1 == 0) | (e3 == 0)] = 10 # one of the principal strains is exactly zero (rare case)
    return signs


sign_map = classify_signs(e1, e3)   # shape (N, 20)


# --------------------------------------------------
# 3. Compress thickness regions
# --------------------------------------------------

def compress_regions(sign_row):
    regions = [sign_row[0]]
    for val in sign_row[1:]:
        if val != regions[-1]:
            regions.append(val)
    return tuple(regions)


# --------------------------------------------------
# 4. Define the 9 regimes (from your image)
# --------------------------------------------------

regime_dict = {

    # 1: fully compression (--)
    (3,): 1,

    # 2: -- → +-
    (1, 3): 2,
    (3, 1): 2,

    # 3: fully tension (++)
    (1,): 3,

    # 4: -- → +-
    (3, 2): 4,
    (2, 3): 4,

    # 5: pure +- (bending dominant)
    (2,): 5,

    # 6: +- → ++
    (1, 2): 6,
    (2, 1): 6,

    # 7: +- → -- → +-
    (2, 3, 2): 7,

    # 8: ++ → +- → --
    (3, 2, 1): 8,
    (1, 2, 3): 8,

    # 9: +- → ++ → +-
    (2, 1, 2): 9,
}


# --------------------------------------------------
# 5. Assign regime per sample
# --------------------------------------------------

N = sign_map.shape[0]
regime_ids = np.zeros(N, dtype=int)

for i in range(N):
    regions = compress_regions(sign_map[i])
    regime_ids[i] = regime_dict.get(regions, 0)   # 0 = not classified




# --------------------------------------------------
# 6. Inspect critical cases (all 10s or unclassified)
# --------------------------------------------------

## Remove samples where all 20 layers are exactly zero (sign_map = 10)
mask_keep = ~(np.all(sign_map == 10, axis=1))
data_e_princ = data_e_princ[mask_keep]
data_eps_g   = data_eps_g[mask_keep]
data_sig_g   = data_sig_g[mask_keep]
regime_ids   = regime_ids[mask_keep]
sign_map     = sign_map[mask_keep]
th           = th[mask_keep]
data_e       = data_e[mask_keep]
n_removed = np.sum(~mask_keep)
print("Removed samples (all 10):", n_removed)

# inspect problematic cases (that are not classified into any of the 9 regimes)
n_unknown = np.sum(regime_ids == 0)
print(f"Unclassified samples: {n_unknown} / {N}")
if n_unknown > 0:
    idx = np.where(regime_ids == 0)[0]

    print("Full sign_map for unknown cases:")
    for i in idx:
        print(f"Sample {i}: {sign_map[i]}")



# --------------------------------------------------
# 7. Now you can use regime_ids for plotting
# --------------------------------------------------

if START_FROM_STEP <= 7:
    plot_3D_data(
        save_data_path,
        filename='sig_g',
        regime_ids=regime_ids,
        data=data_sig_g,
        plotname='scatter_sig_g_regime',
        n_every=1
    )

    plot_3D_data(
        save_data_path,
        filename='eps_g',
        regime_ids=regime_ids,
        data=data_eps_g,
        plotname='scatter_eps_g_regime',
        n_every=1
    )


# --------------------------------------------------
# 8. Plotting regime-wise distributions
# --------------------------------------------------

regime_order = [1, 4, 7, 2, 5, 8, 3, 6, 9]

# --------------------------------------------------
# 8a. Theta change intensity summary for coloring
# --------------------------------------------------

# unwrap + convert
th_unwrapped = np.unwrap(th, axis=1)
th_deg = np.rad2deg(th_unwrapped)

# max difference over thickness #flag check theta
theta_range = np.max(th_deg, axis=1) - np.min(th_deg, axis=1)


# --------------------------------------------------
# 8b. Regime-wise 4-panel overview (eps, chi, n, m)
# --------------------------------------------------

if START_FROM_STEP <= 8:
    fig = plt.figure(figsize=(32, 24))
    outer = fig.add_gridspec(3, 3, wspace=0.18, hspace=0.28)

    for panel_idx, regime_id in enumerate(regime_order):
        row, col = divmod(panel_idx, 3)
        inner = outer[row, col].subgridspec(2, 2, wspace=0.08, hspace=0.20)

        ax_eps = fig.add_subplot(inner[0, 0], projection='3d')
        ax_chi = fig.add_subplot(inner[0, 1], projection='3d')
        ax_n = fig.add_subplot(inner[1, 0], projection='3d')
        ax_m = fig.add_subplot(inner[1, 1], projection='3d')

        mask = regime_ids == regime_id
        data_eps_r = data_eps_g[mask]
        data_sig_r = data_sig_g[mask]

        # faint gray context: all points from the full dataset to show where this regime sits
        ax_eps.scatter(
            data_eps_g[:, 0], data_eps_g[:, 1], data_eps_g[:, 2],
            s=1, alpha=0.01, color='lightgray', depthshade=False, zorder=1
        )
        ax_chi.scatter(
            data_eps_g[:, 3], data_eps_g[:, 4], data_eps_g[:, 5],
            s=1, alpha=0.01, color='lightgray', depthshade=False, zorder=1
        )
        ax_n.scatter(
            data_sig_g[:, 0], data_sig_g[:, 1], data_sig_g[:, 2],
            s=1, alpha=0.01, color='lightgray', depthshade=False, zorder=1
        )
        ax_m.scatter(
            data_sig_g[:, 3], data_sig_g[:, 4], data_sig_g[:, 5],
            s=1, alpha=0.01, color='lightgray', depthshade=False, zorder=1
        )

        sc_eps = ax_eps.scatter(
            data_eps_r[:, 0], data_eps_r[:, 1], data_eps_r[:, 2],
            s=2, alpha=0.35, c=theta_range[mask], cmap='coolwarm', vmin=0, vmax=180, zorder=2
        )
        sc_chi = ax_chi.scatter(
            data_eps_r[:, 3], data_eps_r[:, 4], data_eps_r[:, 5],
            s=2, alpha=0.35, c=theta_range[mask], cmap='coolwarm', vmin=0, vmax=180, zorder=2
        )
        sc_n = ax_n.scatter(
            data_sig_r[:, 0], data_sig_r[:, 1], data_sig_r[:, 2],
            s=2, alpha=0.35, c=theta_range[mask], cmap='coolwarm', vmin=0, vmax=180, zorder=2
        )
        sc_m = ax_m.scatter(
            data_sig_r[:, 3], data_sig_r[:, 4], data_sig_r[:, 5],
            s=2, alpha=0.35, c=theta_range[mask], cmap='coolwarm', vmin=0, vmax=180, zorder=2
        )

        fig.colorbar(sc_eps, ax=ax_eps, pad=0.08, shrink=0.7).set_label('Theta range [deg]')
        fig.colorbar(sc_chi, ax=ax_chi, pad=0.08, shrink=0.7).set_label('Theta range [deg]')
        fig.colorbar(sc_n, ax=ax_n, pad=0.08, shrink=0.7).set_label('Theta range [deg]')
        fig.colorbar(sc_m, ax=ax_m, pad=0.08, shrink=0.7).set_label('Theta range [deg]')

        figure_formatting(ax_eps, 0, 'eps_g')
        figure_formatting(ax_chi, 1, 'eps_g')
        figure_formatting(ax_n, 0, 'sig_g')
        figure_formatting(ax_m, 1, 'sig_g')

        ax_eps.set_title(f'eps — regime {regime_id}')
        ax_chi.set_title(f'chi — regime {regime_id}')
        ax_n.set_title(f'n — regime {regime_id}')
        ax_m.set_title(f'm — regime {regime_id}')

    output_name = 'regimewise_4panel_overview.png'
    fig.savefig(os.path.join(plot_dir, output_name), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_name} to {plot_dir}')

# --------------------------------------------------
# 9. Regime characteristics summary (including th)
# --------------------------------------------------

if START_FROM_STEP <= 9:
    print('\n=== Regime characteristics summary ===\n')

    header = (
        "Reg | N | eps_mean | chi_mean | n_mean | m_mean | "
        "th | eps_dom | load_dom | chi_dom | mode | th_span | sign"
    )
    print(header)
    print('-' * len(header))

    for regime_id in regime_order:
        mask = regime_ids == regime_id
        n = int(np.sum(mask))
        if n == 0:
            continue

        # --- means ---
        eps_mean = np.mean(data_eps_g[mask, :3], axis=0)
        chi_mean = np.mean(data_eps_g[mask, 3:6], axis=0)
        n_mean   = np.mean(data_sig_g[mask, :3], axis=0)
        m_mean   = np.mean(data_sig_g[mask, 3:6], axis=0)

        th_mean = np.mean(th[mask])
        th_min  = np.min(th[mask])
        th_max  = np.max(th[mask])

        # --- dominance ---
        comps = {
            'eps_x': eps_mean[0], 'eps_y': eps_mean[1], 'eps_xy': eps_mean[2],
            'n_x': n_mean[0], 'n_y': n_mean[1], 'n_xy': n_mean[2],
            'm_x': m_mean[0], 'm_y': m_mean[1], 'm_xy': m_mean[2],
            'chi_x': chi_mean[0], 'chi_y': chi_mean[1], 'chi_xy': chi_mean[2],
        }

        dominant_eps = max(['eps_x','eps_y','eps_xy'], key=lambda k: abs(comps[k]))
        dominant_loading = max(['n_x','n_y','n_xy','m_x','m_y','m_xy'], key=lambda k: abs(comps[k]))
        dominant_chi = max(['chi_x','chi_y','chi_xy'], key=lambda k: abs(comps[k]))

        # --- simplified sign pattern (only membrane + bending)
        sign_eps = ''.join('+' if val >= 0 else '-' for val in eps_mean)
        sign_n   = ''.join('+' if val >= 0 else '-' for val in n_mean)
        sign_m   = ''.join('+' if val >= 0 else '-' for val in m_mean)

        sign_pattern = f"E:{sign_eps} N:{sign_n} M:{sign_m}"

        # --- mode classification ---
        norm_eps = np.linalg.norm(eps_mean)
        norm_chi = np.linalg.norm(chi_mean)

        if norm_eps > 2 * norm_chi:
            mode = "membrane"
        elif norm_chi > 2 * norm_eps:
            mode = "bending"
        else:
            mode = "mixed"

        # --- thickness spread ---
        th_span = th_max - th_min

        # --- print ---
        print(
            f"{regime_id:3d} | {n:5d} | "
            f"[{eps_mean[0]:+.2e},{eps_mean[1]:+.2e},{eps_mean[2]:+.2e}] | "
            f"[{chi_mean[0]:+.2e},{chi_mean[1]:+.2e},{chi_mean[2]:+.2e}] | "
            f"[{n_mean[0]:+.2e},{n_mean[1]:+.2e},{n_mean[2]:+.2e}] | "
            f"[{m_mean[0]:+.2e},{m_mean[1]:+.2e},{m_mean[2]:+.2e}] | "
            f"{th_mean:.2e} | "
            f"{dominant_eps:6s} | {dominant_loading:6s} | {dominant_chi:6s} | "
            f"{mode:8s} | "
            f"{th_span:.2e} | "
            f"{sign_pattern}"
        )

    # --------------------------------------------------
    # 9a. Theta rotation per regime with correlation
    # --------------------------------------------------

    print("\nTheta rotation per regime:")

    for r in range(1, 10):
        mask = regime_ids == r
        if np.sum(mask) == 0:
            continue

        mean_val = np.mean(theta_range[mask])
        max_val  = np.max(theta_range[mask])
        std_val  = np.std(theta_range[mask])

        # simple correlation between theta range and mean thickness rotation
        x = np.mean(th_deg[mask], axis=1)
        y = theta_range[mask]
        corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan

        print(
            f"Regime {r}: mean = {mean_val:.2f}°, max = {max_val:.2f}°, "
            f"std = {std_val:.2f}°, corr(theta_mean, theta_range) = {corr:.2f}"
        )

    # --------------------------------------------------
    # 9b. Boxplots for theta range variation across regimes
    # --------------------------------------------------

    fig_bp, ax_bp = plt.subplots(figsize=(14, 6))

    regime_labels = [str(r) for r in regime_order]
    regime_data = [theta_range[regime_ids == r] for r in regime_order]

    # Sort the regime-specific values in ascending regime order (already ensured by regime_order)
    regime_data = [theta_range[regime_ids == r] for r in sorted(regime_order)]
    regime_labels = [str(r) for r in sorted(regime_order)]

    ax_bp.boxplot(regime_data, labels=regime_labels, patch_artist=True)
    ax_bp.set_title('Theta range distribution by regime')
    ax_bp.set_ylabel('Theta range [deg]')
    ax_bp.set_xlabel('Regime')

    output_name = 'theta_range_boxplots.png'
    fig_bp.savefig(os.path.join(plot_dir, output_name), dpi=180, bbox_inches='tight')
    plt.close(fig_bp)
    print(f'Saved {output_name} to {plot_dir}')

# --------------------------------------------------
# 10. Regime-specific analysis: start with regime x
# --------------------------------------------------

PROCESS_ALL_REGIMES = True   # True = loop over all, False = single
REGIME_ID = 4                # used only if PROCESS_ALL_REGIMES = False

if START_FROM_STEP <= 10:
    print('enter loop')
    if PROCESS_ALL_REGIMES:
        regime_list = np.unique(regime_ids)
    else:
        regime_list = [REGIME_ID]
    print(regime_list)

    for regime_id in regime_list:
        mask_regime = regime_ids == regime_id

        if np.any(mask_regime):
            regime_plot_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'plots',
                f'Regime_{regime_id}'
            )
            os.makedirs(regime_plot_dir, exist_ok=True)


            eps_reg = data_eps_g[mask_regime]
            sig_reg = data_sig_g[mask_regime]

            e1_reg = np.ravel(data_e_princ[mask_regime, :, 0])
            e3_reg = np.ravel(data_e_princ[mask_regime, :, 1])
            th_reg_deg = np.ravel(th_deg[mask_regime])

            # Top: e, chi, e1, e3, th
            fig_reg, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10))

            labels_top = [
                'e_x', 'e_y', 'e_xy',
                'chi_x', 'chi_y', 'chi_xy',
                'e1', 'e3', 'th'
            ]
            def normalize_component(x):
                x = np.asarray(x)
                max_abs = np.max(np.abs(x))
                return x / max_abs if max_abs > 0 else x

            data_top = [
                normalize_component(eps_reg[:, 0]), normalize_component(eps_reg[:, 1]), normalize_component(eps_reg[:, 2]),
                normalize_component(eps_reg[:, 3]), normalize_component(eps_reg[:, 4]), normalize_component(eps_reg[:, 5]),
                normalize_component(e1_reg), normalize_component(e3_reg), normalize_component(th_reg_deg)
            ]

            ax_top.boxplot(data_top, labels=labels_top, patch_artist=True)
            ax_top.set_title(f'Regime {regime_id}: boxplots for e, chi, e1, e3, th')
            ax_top.set_ylabel('Value')
            ax_top.grid(True, axis='y', alpha=0.2)

            # Bottom: n and m components
            labels_bottom = ['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy']
            data_bottom = [
                normalize_component(sig_reg[:, 0]), normalize_component(sig_reg[:, 1]), normalize_component(sig_reg[:, 2]),
                normalize_component(sig_reg[:, 3]), normalize_component(sig_reg[:, 4]), normalize_component(sig_reg[:, 5]),
            ]

            ax_bottom.boxplot(data_bottom, labels=labels_bottom, patch_artist=True)
            ax_bottom.set_title(f'Regime {regime_id}: boxplots for n and m components')
            ax_bottom.set_ylabel('Value')
            ax_bottom.grid(True, axis='y', alpha=0.2)

            fig_reg.tight_layout()
            output_reg = os.path.join(regime_plot_dir, f'regime_{regime_id}_boxplots.png')
            fig_reg.savefig(output_reg, dpi=180, bbox_inches='tight')
            plt.close(fig_reg)
            print(f'Saved regime {regime_id} boxplots to {output_reg}')
        else:
            print(f'No samples found in regime {regime_id}; skipping regime-specific boxplots.')

        # --------------------------------------------------
        # 10a. 2D projection for points with nxy and mxy near zero
        # --------------------------------------------------
        
        # Used to filter points with nxy and mxy near zero for better visualization of the regime characteristics.
        # nxy_tol = 5000000
        # mxy_tol = 5000000

        # mask_box = (
        #     mask_regime &
        #     (np.abs(data_sig_g[:, 2]) <= nxy_tol) &
        #     (np.abs(data_sig_g[:, 5]) <= mxy_tol)
        # )

        mask_box = mask_regime

        # -----------------------------
        # 2D projection plot
        # -----------------------------

        if np.any(mask_box):
            sel = data_sig_g[mask_box]

            fig_box, axes = plt.subplots(1, 2, figsize=(14, 6))

            # -------------------------
            # n_x vs n_y (colour = nxy)
            # -------------------------
            vmax_n = np.max(np.abs(sel[:, 2]))
            sc1 = axes[0].scatter(
                sel[:, 0], sel[:, 1],
                c=sel[:, 2],          # nxy
                cmap='viridis',
                vmin=-vmax_n, vmax=vmax_n,
                s=10, alpha=0.7
            )
            axes[0].set_xlabel('n_x')
            axes[0].set_ylabel('n_y')
            axes[0].set_title('n_x vs n_y (colour = n_xy)')
            axes[0].grid(True, alpha=0.2)
            fig_box.colorbar(sc1, ax=axes[0], label='n_xy')

            # -------------------------
            # m_x vs m_y (colour = mxy)
            # -------------------------
            vmax_m = np.max(np.abs(sel[:, 5]))
            sc2 = axes[1].scatter(
                sel[:, 3], sel[:, 4],
                c=sel[:, 5],          # mxy
                cmap='plasma',
                vmin=-vmax_m, vmax=vmax_m,
                s=10, alpha=0.7
            )
            axes[1].set_xlabel('m_x')
            axes[1].set_ylabel('m_y')
            axes[1].set_title('m_x vs m_y (colour = m_xy)')
            axes[1].grid(True, alpha=0.2)
            fig_box.colorbar(sc2, ax=axes[1], label='m_xy')

            fig_box.tight_layout()

            output_box = os.path.join(
                regime_plot_dir,
                f'nxy_mxy_near_zero_2d_regime_{regime_id}.png'
            )
            fig_box.savefig(output_box, dpi=180, bbox_inches='tight')
            plt.close(fig_box)

            print(f'Saved near-zero nxy/mxy 2D plot to {output_box}')
        else:
            print('No points found in the selected near-zero nxy/mxy window.')



        # -----------------------------
        # 10b Plot strain profiles for selected points (extrema in nx, ny, mx, my, nxy, mxy)
        # -----------------------------
        filtered = data_sig_g[mask_box]
        filtered_idx = np.where(mask_box)[0]

        if filtered.shape[0] == 0:
            print("No points satisfy the filter condition!")
            continue

        # -----------------------------
        # Indices of extrema (filtered)
        # -----------------------------
        def get_index(col, func):
            local_idx = func(filtered[:, col])
            return filtered_idx[local_idx]

        indices_extrema = {
            "nx_min": get_index(0, np.argmin),
            "nx_max": get_index(0, np.argmax),
            "ny_min": get_index(1, np.argmin),
            "ny_max": get_index(1, np.argmax),
            "mx_min": get_index(3, np.argmin),
            "mx_max": get_index(3, np.argmax),
            "my_min": get_index(4, np.argmin),
            "my_max": get_index(4, np.argmax),
            "nxy_min_f": get_index(2, np.argmin),
            "nxy_max_f": get_index(2, np.argmax),
            "mxy_min_f": get_index(5, np.argmin),
            "mxy_max_f": get_index(5, np.argmax),
        }

        # -----------------------------
        # z coordinate
        # -----------------------------
        z_mm = (np.arange(data_e.shape[1]) - (data_e.shape[1] - 1) / 2.0) * (350.0 / data_e.shape[1])

        print('\n=== Selected layerwise strain diagnostics (engineering plots) ===')

        for name, idx in indices_extrema.items():

            ex  = data_e[idx, :, 0]
            ey  = data_e[idx, :, 1]
            exy = data_e[idx, :, 2]
            e1  = data_e_princ[idx, :, 0]
            e3  = data_e_princ[idx, :, 1]
            th  = th_deg[idx]


            fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)

            # -----------------------------
            # ex
            # -----------------------------
            axes[0,0].plot(ex, z_mm, color='black')
            axes[0,0].fill_betweenx(z_mm, 0, ex, color='gray', alpha=0.5)
            axes[0,0].set_title(r'$\epsilon_x$')
            axes[0,0].grid(True, alpha=0.2)

            # -----------------------------
            # ey
            # -----------------------------
            axes[0,1].plot(ey, z_mm, color='black')
            axes[0,1].fill_betweenx(z_mm, 0, ey, color='gray', alpha=0.5)
            axes[0,1].set_title(r'$\epsilon_y$')
            axes[0,1].grid(True, alpha=0.2)

            # -----------------------------
            # exy
            # -----------------------------
            axes[0,2].plot(exy, z_mm, color='black')
            axes[0,2].fill_betweenx(z_mm, 0, exy, color='gray', alpha=0.5)
            axes[0,2].set_title(r'$\epsilon_{xy}$')
            axes[0,2].grid(True, alpha=0.2)

            
        
            # -----------------------------
            # use existing sign_map
            # -----------------------------
            zones = sign_map[idx]   # assumed shape (nz,)

            axes[1,0].set_title('Cross-section zones + θ')

            # -----------------------------
            # draw zones (background)
            # -----------------------------
            colors = {
                1: 'lightgray',       # ++
                2: 'darkgray',    # +-
                3: 'dimgray'   # --
            }

            for val in [1, 2, 3]:
                mask = zones == val
                if np.any(mask):
                    axes[1,0].fill_betweenx(
                        z_mm, 0, 1,
                        where=mask,
                        color=colors[val],
                        step='pre'
                    )

            axes[1,0].set_xlim(0, 1)
            axes[1,0].set_xticks([])
            axes[1,0].grid(True, axis='y', alpha=0.2)


            # -----------------------------
            # overlay theta (second x-axis)
            # -----------------------------
            ax_theta = axes[1,0].twiny()
            ax_theta.plot(th, z_mm, color='black', linewidth=1.5)

            ax_theta.set_xlabel(r'$\theta$ [deg]')
            ax_theta.axvline(0, color='black', linestyle='--', linewidth=0.8)



            # -----------------------------
            # e1
            # -----------------------------
            axes[1,1].plot(e1, z_mm, color='black', linewidth=2)
            axes[1,1].fill_betweenx(z_mm, 0, e1, color='gray', alpha=0.5)
            axes[1,1].set_title(r'$\epsilon_1$')
            axes[1,1].grid(True, alpha=0.2)

            # -----------------------------
            # e3
            # -----------------------------
            axes[1,2].plot(e3, z_mm, color='black', linewidth=2)
            axes[1,2].fill_betweenx(z_mm, 0, e3, color='gray', alpha=0.5)
            axes[1,2].set_title(r'$\epsilon_3$')
            axes[1,2].grid(True, alpha=0.2)


            # -----------------------------
            # shared labels
            # -----------------------------
            for ax in axes[:,0]:
                ax.set_ylabel('z [mm]')
            
            def fmt(v):
                return f"{v:,.3f}".replace(",", "'")

            fig.suptitle(
                f'{name} (idx={idx}) | nxy={data_sig_g[idx,2]:.3e}, mxy={data_sig_g[idx,5]:.3e}, regime={regime_ids[idx]}'
            )
            
            fig.suptitle(
                f'{name} (idx={idx})\n'
                f'nx={fmt(data_sig_g[idx,0])}, '
                f'ny={fmt(data_sig_g[idx,1])}, '
                f'nxy={fmt(data_sig_g[idx,2])} | '
                f'mx={fmt(data_sig_g[idx,3])}, '
                f'my={fmt(data_sig_g[idx,4])}, '
                f'mxy={fmt(data_sig_g[idx,5])}\n'
                f'regime={regime_ids[idx]}'
            )

            fig.tight_layout()

            output_file = os.path.join(regime_plot_dir, f'cs_profile_{name}_idx{idx}_regime_{regime_id}.png')
            fig.savefig(output_file, dpi=180, bbox_inches='tight')
            plt.close(fig)

            print(f'Saved: {output_file}')

