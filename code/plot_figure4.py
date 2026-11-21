import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
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
        axis="both",
        which="major",
        direction="out",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
        length=4.0,
        width=0.85,
        pad=1.8,
        labelsize=5.2,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        top=True,
        bottom=True,
        length=2.2,
        width=0.65,
    )
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.85)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate main Figure 4.")
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.repo_root.resolve()
    source = root / "data" / "derived" / "Figure4_source_fields_0p5deg.npz"
    land_mask = root / "code" / "haf_core" / "released" / "land_mask_0p5deg.npz"
    cartopy_data = root / "code" / "cartopy_data"
    output = args.output_dir.resolve() if args.output_dir else root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    cartopy.config["data_dir"] = str(cartopy_data)

    with np.load(source) as data:
        lon = np.asarray(data["lon"], float)
        lat = np.asarray(data["lat"], float)
        fields = {
            (scenario, period): np.asarray(
                data[f"delta_total_{scenario}_{period}"], float
            )
            for scenario in SCENARIOS
            for period in PERIODS
        }

    natural_earth_land = np.load(land_mask)["mask"].astype(bool)
    ice_free_land = natural_earth_land & (lat[:, None] > -60.0)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    panel_means = {}
    for key, field in fields.items():
        valid = ice_free_land & np.isfinite(field)
        panel_means[key] = float(np.average(field[valid], weights=area[valid]))

    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )

    levels = np.arange(-4.0, 4.01, 0.5)
    diverging = LinearSegmentedColormap.from_list(
        "delta_chs",
        [
            "#2166ac",
            "#67a9cf",
            "#d1e5f0",
            "#f7f7f7",
            "#fddbc7",
            "#ef8a62",
            "#b2182b",
        ],
        N=len(levels) - 1,
    )
    norm = BoundaryNorm(levels, diverging.N, clip=False)

    projection = ccrs.PlateCarree(central_longitude=-90)
    transform = ccrs.PlateCarree()

    fig = plt.figure(figsize=(7.205, 5.78))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.07,
        right=0.985,
        top=0.965,
        bottom=0.125,
        wspace=0.060,
        hspace=0.30,
    )

    axes = np.empty((3, 2), dtype=object)
    mappable = None
    for row, scenario in enumerate(SCENARIOS):
        for col, period in enumerate(PERIODS):
            ax = fig.add_subplot(grid[row, col], projection=projection)
            axes[row, col] = ax
            field = fields[(scenario, period)]
            mappable = ax.contourf(
                lon,
                lat,
                field,
                levels=levels,
                cmap=diverging,
                norm=norm,
                extend="both",
                transform=transform,
                antialiased=False,
            )
            ax.coastlines(resolution="110m", linewidth=0.48, color="black")
            style_map_axis(ax)
            ax.text(
                0.0,
                1.032,
                chr(ord("a") + row + 3 * col),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                clip_on=False,
            )
            ax.text(
                0.5,
                1.032,
                f"{panel_means[(scenario, period)]:+.3f}".replace("-", "−"),
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="normal",
                clip_on=False,
            )

    # Close the actual visible gap left by the fixed 2:1 geographic aspect.
    for row in range(3):
        left_pos = axes[row, 0].get_position()
        right_pos = axes[row, 1].get_position()
        axes[row, 0].set_position(left_pos)
        axes[row, 1].set_position(right_pos.translated(-0.042, 0.0))

    fig.canvas.draw()
    ticks = np.arange(-4.0, 4.1, 2.0)
    for col in range(2):
        bottom_position = axes[2, col].get_position()
        cax = fig.add_axes(
            [
                bottom_position.x0 + 0.055 * bottom_position.width,
                bottom_position.y0 - 0.046,
                0.89 * bottom_position.width,
                0.012,
            ]
        )
        cb = fig.colorbar(
            mappable,
            cax=cax,
            orientation="horizontal",
            extend="both",
            ticks=ticks,
            spacing="uniform",
            drawedges=True,
        )
        cb.ax.tick_params(
            axis="x", which="major", direction="out", length=3.0,
            width=0.75, pad=2.0
        )
        cb.outline.set_edgecolor("black")
        cb.outline.set_linewidth(0.75)
        cb.dividers.set_color("black")
        cb.dividers.set_linewidth(0.35)
        cb.set_label("")

    png = output / "Figure4.png"
    pdf = output / "Figure4.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
