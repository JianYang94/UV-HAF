import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator

import cartopy
import cartopy.crs as ccrs


SCENARIOS = ("ssp126", "ssp245", "ssp585")


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
        labeltop=False, labelright=False, length=4.0, width=0.85,
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
    parser = argparse.ArgumentParser(description="Regenerate main Figure 5.")
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.repo_root.resolve()
    source = root / "data" / "derived" / "Figure5_source_fields_0p5deg.npz"
    cartopy_data = root / "code" / "cartopy_data"
    output = args.output_dir.resolve() if args.output_dir else root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    cartopy.config["data_dir"] = str(cartopy_data)

    with np.load(source) as data:
        lon = np.asarray(data["lon"], float)
        lat = np.asarray(data["lat"], float)
        emergence = {
            scenario: np.asarray(data[f"emergence7_{scenario}"], float)
            for scenario in SCENARIOS
        }
        shift = {
            scenario: np.asarray(data[f"shift_{scenario}"], float)
            for scenario in SCENARIOS
        }
        haf7 = {
            scenario: float(data[f"haf7_{scenario}"])
            for scenario in SCENARIOS
        }
        delta_haf = {
            scenario: float(data[f"delta_haf_{scenario}"])
            for scenario in SCENARIOS
        }

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

    year_levels = np.arange(2015.0, 2100.1, 5.0)
    year_cmap = plt.get_cmap("cividis", len(year_levels) - 1)
    year_norm = BoundaryNorm(year_levels, year_cmap.N, clip=False)

    shift_levels = np.arange(-30.0, 30.01, 5.0)
    shift_cmap = LinearSegmentedColormap.from_list(
        "crossing_shift",
        ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"],
        N=len(shift_levels) - 1,
    )
    shift_norm = BoundaryNorm(shift_levels, shift_cmap.N, clip=False)

    baseline_color = "#3d3d3d"
    never_color = "#d9d9d9"
    projection = ccrs.PlateCarree(central_longitude=-90)
    transform = ccrs.PlateCarree()

    fig = plt.figure(figsize=(7.205, 5.90))
    grid = fig.add_gridspec(
        3, 2, left=0.07, right=0.985, top=0.965, bottom=0.145,
        wspace=0.060, hspace=0.30,
    )

    axes = np.empty((3, 2), dtype=object)
    year_mappable = None
    shift_mappable = None
    for row, scenario in enumerate(SCENARIOS):
        for col in range(2):
            ax = fig.add_subplot(grid[row, col], projection=projection)
            axes[row, col] = ax

            if col == 0:
                field = emergence[scenario]
                crossing = np.where(
                    (field >= 2015) & (field <= 2100), field, np.nan
                )
                year_mappable = ax.pcolormesh(
                    lon, lat, crossing, cmap=year_cmap, norm=year_norm,
                    transform=transform, shading="auto", rasterized=True,
                )
                baseline = np.where(field == 2014, 1.0, np.nan)
                never = np.where(field == 2101, 1.0, np.nan)
                ax.pcolormesh(
                    lon, lat, baseline, cmap=ListedColormap([baseline_color]),
                    vmin=0.5, vmax=1.5, transform=transform, shading="auto",
                    rasterized=True,
                )
                ax.pcolormesh(
                    lon, lat, never, cmap=ListedColormap([never_color]),
                    vmin=0.5, vmax=1.5, transform=transform, shading="auto",
                    rasterized=True,
                )
                value = f"{haf7[scenario]:+.3f}"
            else:
                field = shift[scenario]
                shift_mappable = ax.pcolormesh(
                    lon, lat, field, cmap=shift_cmap, norm=shift_norm,
                    transform=transform, shading="auto", rasterized=True,
                )
                value = f"{delta_haf[scenario]:+.3f}".replace("-", "−")

            ax.coastlines(resolution="110m", linewidth=0.48, color="black")
            style_map_axis(ax)
            ax.text(
                0.0, 1.032, chr(ord("a") + row + 3 * col),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
                fontweight="bold", clip_on=False,
            )
            ax.text(
                0.5, 1.032, value, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.0, fontweight="normal", clip_on=False,
            )

    # Close the actual visible gap left by the fixed 2:1 geographic aspect.
    for row in range(3):
        left_pos = axes[row, 0].get_position()
        right_pos = axes[row, 1].get_position()
        axes[row, 0].set_position(left_pos)
        axes[row, 1].set_position(right_pos.translated(-0.042, 0.0))

    fig.canvas.draw()
    colorbar_axes = []
    for col, mappable in enumerate((year_mappable, shift_mappable)):
        bottom_position = axes[2, col].get_position()
        cax = fig.add_axes(
            [
                bottom_position.x0 + 0.055 * bottom_position.width,
                bottom_position.y0 - 0.046,
                0.89 * bottom_position.width,
                0.012,
            ]
        )
        colorbar_axes.append(cax)
        if col == 0:
            cb = fig.colorbar(
                mappable, cax=cax, orientation="horizontal",
                ticks=[2020, 2040, 2060, 2080, 2100], spacing="uniform",
                drawedges=True,
            )
        else:
            cb = fig.colorbar(
                mappable, cax=cax, orientation="horizontal", extend="both",
                ticks=[-30, -15, 0, 15, 30], spacing="uniform",
                drawedges=True,
            )
            cb.ax.set_xticklabels(["−30", "−15", "0", "15", "30"])
        cb.ax.tick_params(
            axis="x", which="major", direction="out", length=3.0,
            width=0.75, pad=2.0,
        )
        cb.outline.set_edgecolor("black")
        cb.outline.set_linewidth(0.75)
        cb.dividers.set_color("black")
        cb.dividers.set_linewidth(0.35)
        cb.set_label("")

    left_cax = colorbar_axes[0].get_position()
    fig.legend(
        handles=[
            Patch(facecolor=baseline_color, edgecolor="black", linewidth=0.35,
                  label="Above threshold in baseline"),
            Patch(facecolor=never_color, edgecolor="black", linewidth=0.35,
                  label="No crossing by 2100"),
        ],
        loc="upper center",
        bbox_to_anchor=(left_cax.x0 + left_cax.width / 2, left_cax.y0 - 0.019),
        ncol=2,
        frameon=False,
        fontsize=5.2,
        handlelength=1.25,
        handleheight=0.65,
        columnspacing=1.15,
        borderaxespad=0,
    )

    png = output / "Figure5.png"
    pdf = output / "Figure5.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
