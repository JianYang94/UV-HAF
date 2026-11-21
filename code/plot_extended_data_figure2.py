import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.ticker import FixedLocator

import cartopy
import cartopy.crs as ccrs


SCENARIOS = ("ssp126", "ssp245", "ssp585")
PERIODS = ("mid", "late")


def longitude_label(x):
    lon = ((x - 90.0 + 180.0) % 360.0) - 180.0
    if np.isclose(abs(lon), 180.0):
        return "180°"
    if np.isclose(lon, 0.0):
        return "0°"
    value = int(round(abs(lon)))
    return f"{value}°{'E' if lon > 0 else 'W'}"


def latitude_label(y):
    if np.isclose(y, 0.0):
        return "0°"
    value = int(round(abs(y)))
    return f"{value}°{'N' if y > 0 else 'S'}"


def style_map_axis(ax):
    ax.set_global()
    major_x = np.arange(-180, 181, 30)
    minor_x = np.arange(-180, 181, 10)
    major_y = np.arange(-90, 91, 30)
    ax.set_xticks(major_x)
    ax.set_yticks(major_y)
    ax.xaxis.set_minor_locator(FixedLocator(minor_x))
    ax.set_xticklabels([longitude_label(x) for x in major_x])
    ax.set_yticklabels([latitude_label(y) for y in major_y])
    ax.tick_params(
        axis="both", which="major", direction="out", top=True, right=True,
        labeltop=False, labelright=False, length=3.4, width=0.78,
        pad=1.8, labelsize=5.2,
    )
    ax.tick_params(
        axis="x", which="minor", direction="out", top=True, bottom=True,
        length=2.2, width=0.65,
    )
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.85)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate Extended Data Figure 2.")
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def add_colorbar(
    fig, mappable, rect, ticks, extend="neither", labels=None, drawedges=True
):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(
        mappable, cax=cax, orientation="horizontal", ticks=ticks,
        extend=extend, spacing="uniform", drawedges=drawedges,
    )
    if labels is not None:
        cb.ax.set_xticklabels(labels)
    cb.ax.tick_params(
        axis="x", which="major", direction="out", length=2.8,
        width=0.75, pad=2.0, labelsize=5.2,
    )
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(0.7)
    if drawedges:
        cb.dividers.set_color("black")
        cb.dividers.set_linewidth(0.3)
    cb.set_label("")
    return cax


def main():
    args = parse_args()
    root = args.repo_root.resolve()
    source = (
        root / "data" / "derived" /
        "Extended_Data_Figure_2_source_fields_0p5deg.npz"
    )
    land_mask_file = (
        root / "code" / "haf_core" / "released" / "land_mask_0p5deg.npz"
    )
    cartopy_data = root / "code" / "cartopy_data"
    output = args.output_dir.resolve() if args.output_dir else root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    cartopy.config["data_dir"] = str(cartopy_data)

    with np.load(source) as data:
        lon = np.asarray(data["lon"], float)
        lat = np.asarray(data["lat"], float)
        chs6 = np.asarray(data["chs6_baseline"], float)
        chs7 = np.asarray(data["chs7_baseline"], float)
        baseline_delta = np.asarray(data["delta_chs_baseline"], float)
        pure_uv = {
            (scenario, period): np.asarray(
                data[f"pure_uv_{scenario}_{period}"], float
            )
            for scenario in SCENARIOS
            for period in PERIODS
        }

    natural_earth_land = np.load(land_mask_file)["mask"].astype(bool)
    ice_free_land = natural_earth_land & (lat[:, None] > -60.0)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    uv_means = {}
    for key, field in pure_uv.items():
        valid = ice_free_land & np.isfinite(field)
        uv_means[key] = float(np.average(field[valid], weights=area[valid]))

    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )

    chs_levels = np.arange(-2.0, 10.01, 1.0)
    chs_cmap = LinearSegmentedColormap.from_list(
        "hazard",
        [
            "#fffdf5", "#f9edc7", "#f6d58b", "#f0b35c", "#e8883d",
            "#d65b30", "#b9342d", "#8d1f38", "#54132c",
        ],
        N=len(chs_levels) - 1,
    )
    chs_norm = BoundaryNorm(chs_levels, chs_cmap.N, clip=False)

    delta_cmap = LinearSegmentedColormap.from_list(
        "baseline_delta", ["#2166ac", "#deebf7"], N=256,
    )
    delta_norm = Normalize(vmin=-0.376, vmax=-0.342, clip=False)

    uv_levels = np.arange(-4.0, 4.01, 0.5)
    uv_cmap = LinearSegmentedColormap.from_list(
        "direct_uv",
        ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7",
         "#fddbc7", "#ef8a62", "#b2182b"],
        N=len(uv_levels) - 1,
    )
    uv_norm = BoundaryNorm(uv_levels, uv_cmap.N, clip=False)

    projection = ccrs.PlateCarree(central_longitude=-90)
    transform = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10.8075, 5.90))
    grid = fig.add_gridspec(
        3, 3, left=0.047, right=0.992, top=0.965, bottom=0.155,
        wspace=0.060, hspace=0.30,
    )

    axes = np.empty((3, 3), dtype=object)
    chs_mappable = None
    delta_mappable = None
    uv_mappable = None

    baseline_fields = (chs6, chs7, baseline_delta)
    baseline_labels = (r"CHS$_6$", r"CHS$_7$", r"$\Delta$CHS")
    for row, (field, label) in enumerate(zip(baseline_fields, baseline_labels)):
        ax = fig.add_subplot(grid[row, 0], projection=projection)
        axes[row, 0] = ax
        display_field = np.where(natural_earth_land, field, np.nan)
        if row < 2:
            chs_mappable = ax.contourf(
                lon, lat, display_field, levels=chs_levels, cmap=chs_cmap,
                norm=chs_norm, extend="both", transform=transform,
                antialiased=False,
            )
        else:
            delta_mappable = ax.pcolormesh(
                lon, lat, display_field, cmap=delta_cmap, norm=delta_norm,
                transform=transform, shading="auto", rasterized=True,
            )
        ax.coastlines(resolution="110m", linewidth=0.42, color="black")
        style_map_axis(ax)
        ax.text(
            0.0, 1.035, chr(ord("a") + row), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.5, fontweight="bold",
            clip_on=False,
        )

    for col, period in enumerate(PERIODS, start=1):
        for row, scenario in enumerate(SCENARIOS):
            ax = fig.add_subplot(grid[row, col], projection=projection)
            axes[row, col] = ax
            field = pure_uv[(scenario, period)]
            display_field = np.where(natural_earth_land, field, np.nan)
            uv_mappable = ax.contourf(
                lon, lat, display_field, levels=uv_levels, cmap=uv_cmap,
                norm=uv_norm, extend="both", transform=transform,
                antialiased=False,
            )
            ax.coastlines(resolution="110m", linewidth=0.42, color="black")
            style_map_axis(ax)
            ax.text(
                0.0, 1.035, chr(ord("a") + row + 3 * col),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
                fontweight="bold", clip_on=False,
            )
            value = f"{uv_means[(scenario, period)]:+.3f}".replace("-", "−")
            ax.text(
                0.5, 1.035, value, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.0, clip_on=False,
            )

    # Keep the first column fixed and move only the following columns left.
    # The offsets preserve the same physical inter-column spacing used in the
    # revised two-column main figures despite this figure's wider canvas.
    for row in range(3):
        middle_pos = axes[row, 1].get_position()
        right_pos = axes[row, 2].get_position()
        axes[row, 1].set_position(middle_pos.translated(-0.028, 0.0))
        axes[row, 2].set_position(right_pos.translated(-0.056, 0.0))

    fig.canvas.draw()
    bottom_left = axes[2, 0].get_position()
    bottom_mid = axes[2, 1].get_position()
    bottom_right = axes[2, 2].get_position()

    add_colorbar(
        fig, chs_mappable,
        [
            bottom_left.x0 + 0.055 * bottom_left.width,
            bottom_left.y0 - 0.083,
            0.89 * bottom_left.width,
            0.012,
        ],
        ticks=[-2, 0, 2, 4, 6, 8, 10], extend="both",
    )
    add_colorbar(
        fig, delta_mappable,
        [
            bottom_left.x0 + 0.055 * bottom_left.width,
            bottom_left.y0 - 0.046,
            0.89 * bottom_left.width,
            0.012,
        ],
        ticks=[-0.376, -0.359, -0.342], extend="both",
        labels=["−0.376", "−0.359", "−0.342"],
        drawedges=False,
    )

    for position in (bottom_mid, bottom_right):
        add_colorbar(
            fig, uv_mappable,
            [
                position.x0 + 0.055 * position.width,
                position.y0 - 0.046,
                0.89 * position.width,
                0.012,
            ],
            ticks=[-4, -2, 0, 2, 4], extend="both",
            labels=["−4", "−2", "0", "2", "4"],
        )

    png = output / "Extended_Data_Figure_2.png"
    pdf = output / "Extended_Data_Figure_2.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
