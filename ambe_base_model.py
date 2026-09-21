# ============================================
# ambe_base_model.py
# Shared base OpenMC model for Am-Be paraffin setup
# ============================================

from pathlib import Path
from datetime import date, datetime

import numpy as np
import matplotlib.pyplot as plt

import openmc


# ------------------------------------------------------------
# Optional Alphanso import
# ------------------------------------------------------------

try:
    import alphanso  # noqa: F401
    HAS_ALPHANSO = True
except Exception:
    HAS_ALPHANSO = False


# ============================================================
# Documented physical source record and normalization
# ============================================================

# Source-specific information transcribed from the historical factory test
# report, source inventory, and laboratory description.  The reported
# uncertainty of both neutron-field values is unknown; no numerical
# uncertainty is inferred from the number of displayed digits.
SOURCE_MANUFACTURER = "The Radiochemical Centre, Amersham"
SOURCE_MODEL = "AMN 24"
SOURCE_SERIAL_NUMBER = "AMN 5000/1365"
SOURCE_RADIONUCLIDE = "Am-241"
SOURCE_NOMINAL_ACTIVITY_CI = 5.0
SOURCE_MATERIAL_DESCRIPTION = "Am2O3 mixed with beryllium metal powder"
SOURCE_ENCAPSULATION = "double steel capsule"
SOURCE_INSTALLATION = "embedded in a paraffin moderator"

# Source-specific factory neutron-emission result.
SOURCE_NEUTRON_EMISSION_REF = 1.2e7  # n/s
SOURCE_NEUTRON_EMISSION_REF_DATE = "1968-07-05"
SOURCE_NEUTRON_EMISSION_STANDARD_UNCERTAINTY = None
SOURCE_NEUTRON_EMISSION_UNCERTAINTY_STATUS = (
    "not reported in the available factory documentation"
)

# Am-241 half-life used only to decay-correct the factory neutron-emission
# result to a selected experimental date.
AM241_HALF_LIFE_Y = 432.6
DAYS_PER_YEAR = 365.2425

# Historical local irradiation-position value.  This is a moderated local
# thermal flux, not the total neutron emission rate of the source.  The
# original detector/activation method, thermal-flux definition, averaging
# volume, calibration, and uncertainty budget are not available.
HISTORICAL_LOCAL_THERMAL_FLUX = 2.3e4  # n/cm2/s
HISTORICAL_LOCAL_THERMAL_FLUX_STANDARD_UNCERTAINTY = None
HISTORICAL_LOCAL_THERMAL_FLUX_UNCERTAINTY_STATUS = (
    "unknown; measurement method and uncertainty were not documented"
)

# The historical description also identifies the 4.439 MeV gamma line but
# does not provide a photon yield or gamma-to-neutron emission ratio.
SOURCE_DOCUMENTED_GAMMA_LINE_MEV = 4.439


def _as_date(value, name="date"):
    """Convert an ISO date string, date, or datetime to ``datetime.date``."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} must use ISO format YYYY-MM-DD; got {value!r}."
            ) from exc
    raise TypeError(
        f"{name} must be an ISO date string, datetime.date, or datetime.datetime."
    )


def decay_correct_neutron_emission(
    evaluation_date,
    reference_emission=SOURCE_NEUTRON_EMISSION_REF,
    reference_date=SOURCE_NEUTRON_EMISSION_REF_DATE,
    half_life_years=AM241_HALF_LIFE_Y,
):
    """
    Decay-correct the source-specific factory neutron-emission result.

    The correction assumes that the Am-Be neutron output follows the Am-241
    alpha activity.  It changes only the central normalization.  It does not
    create an uncertainty for the historical factory measurement.
    """

    evaluation_date = _as_date(evaluation_date, "evaluation_date")
    reference_date = _as_date(reference_date, "reference_date")

    elapsed_years = (
        (evaluation_date - reference_date).days / DAYS_PER_YEAR
    )

    emission = float(reference_emission) * 2.0 ** (
        -elapsed_years / float(half_life_years)
    )

    return emission


def get_source_record(evaluation_date=None):
    """Return documented source metadata and the selected normalization."""

    if evaluation_date is None:
        selected_emission = SOURCE_NEUTRON_EMISSION_REF
        selected_date = SOURCE_NEUTRON_EMISSION_REF_DATE
        decay_corrected = False
        normalization_status = (
            "factory reference-date value; set an experimental date before "
            "final physical normalization"
        )
    else:
        selected_date = _as_date(
            evaluation_date,
            "evaluation_date",
        ).isoformat()
        selected_emission = decay_correct_neutron_emission(selected_date)
        decay_corrected = True
        normalization_status = (
            "factory result decay-corrected to the selected experimental date"
        )

    return {
        "SOURCE_MANUFACTURER": SOURCE_MANUFACTURER,
        "SOURCE_MODEL": SOURCE_MODEL,
        "SOURCE_SERIAL_NUMBER": SOURCE_SERIAL_NUMBER,
        "SOURCE_RADIONUCLIDE": SOURCE_RADIONUCLIDE,
        "SOURCE_NOMINAL_ACTIVITY_CI": SOURCE_NOMINAL_ACTIVITY_CI,
        "SOURCE_MATERIAL_DESCRIPTION": SOURCE_MATERIAL_DESCRIPTION,
        "SOURCE_ENCAPSULATION": SOURCE_ENCAPSULATION,
        "SOURCE_INSTALLATION": SOURCE_INSTALLATION,
        "SOURCE_NEUTRON_EMISSION_REF": SOURCE_NEUTRON_EMISSION_REF,
        "SOURCE_NEUTRON_EMISSION_REF_DATE": SOURCE_NEUTRON_EMISSION_REF_DATE,
        "SOURCE_NEUTRON_EMISSION_STANDARD_UNCERTAINTY": (
            SOURCE_NEUTRON_EMISSION_STANDARD_UNCERTAINTY
        ),
        "SOURCE_NEUTRON_EMISSION_UNCERTAINTY_STATUS": (
            SOURCE_NEUTRON_EMISSION_UNCERTAINTY_STATUS
        ),
        "AM241_HALF_LIFE_Y": AM241_HALF_LIFE_Y,
        "SOURCE_EVALUATION_DATE": selected_date,
        "SOURCE_NEUTRON_EMISSION": selected_emission,
        "SOURCE_EMISSION_DECAY_CORRECTED": decay_corrected,
        "SOURCE_NORMALIZATION_STATUS": normalization_status,
        "HISTORICAL_LOCAL_THERMAL_FLUX": (
            HISTORICAL_LOCAL_THERMAL_FLUX
        ),
        "HISTORICAL_LOCAL_THERMAL_FLUX_STANDARD_UNCERTAINTY": (
            HISTORICAL_LOCAL_THERMAL_FLUX_STANDARD_UNCERTAINTY
        ),
        "HISTORICAL_LOCAL_THERMAL_FLUX_UNCERTAINTY_STATUS": (
            HISTORICAL_LOCAL_THERMAL_FLUX_UNCERTAINTY_STATUS
        ),
        "SOURCE_DOCUMENTED_GAMMA_LINE_MEV": (
            SOURCE_DOCUMENTED_GAMMA_LINE_MEV
        ),
    }


# ============================================================
# 1) Am-Be neutron energy distribution
# ============================================================

def build_ambe_energy_distribution(make_plot=False, verbose=False):
    """
    Build the Am-Be neutron energy distribution used by the OpenMC source.

    The function first tries to calculate the Am-Be neutron spectrum with
    ALPHANSO. If ALPHANSO fails, it falls back to a built-in approximate
    Am-Be histogram.

    Parameters
    ----------
    make_plot : bool
        If True, plot the neutron emission spectrum.

    verbose : bool
        If True, print information about the spectrum.

    Returns
    -------
    energy_dist : openmc.stats.Discrete
        OpenMC neutron energy distribution in eV.
    """

    def ambe_spectrum_fallback_histogram():
        """
        Return fallback Am-Be spectrum:
        energy bin edges in MeV and normalized bin probabilities.

        This is only used if ALPHANSO fails.
        """
        energy_edges_mev = np.array([
            0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0,
            2.5, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0
        ], dtype=float)

        # One value per bin: len = len(edges) - 1
        spectrum = np.array([
            0.005, 0.01, 0.02, 0.04, 0.07, 0.11, 0.13, 0.13,
            0.12, 0.10, 0.08, 0.06, 0.045, 0.03, 0.02, 0.01
        ], dtype=float)

        spectrum = np.clip(spectrum, 0.0, None)
        spectrum /= spectrum.sum()

        return energy_edges_mev, spectrum

    try:
        if not HAS_ALPHANSO:
            raise RuntimeError("ALPHANSO is not available.")

        from alphanso.transport import Transport

        config = {
            "name": "AMN 24 Am-Be spectrum approximation",
            "calc_type": "homogeneous",
            # These legacy mass-fraction assumptions are used to construct
            # the spectral shape only.  The available AMN 24 documentation
            # does not establish this mixture ratio, active mixture mass, or
            # a yield uncertainty, so the ALPHANSO yield is not used as the
            # absolute source normalization.
            "matdef": {
                "Am-241": 0.673,
                "Be-9": 0.327,
            },
            # [start_MeV, stop_MeV, number_of_edges]
            "neutron_energy_bins": [0.0, 12.0, 121],
        }

        if verbose:
            print("Running ALPHANSO calculation...")

        results = Transport.calculate(config)

        an_yield = float(results["an_yield"])

        energy_edges = np.asarray(
            results["neutron_energy_bins"],
            dtype=float
        )

        normalized_spectrum = np.asarray(
            results["an_spectrum"],
            dtype=float
        )

        # Sometimes neutron_energy_bins can be returned as
        # [start, stop, number_of_edges] instead of full edges.
        if (
            energy_edges.size == 3
            and normalized_spectrum.size != energy_edges.size - 1
        ):
            e_start = float(energy_edges[0])
            e_stop = float(energy_edges[1])
            n_edges = int(energy_edges[2])
            energy_edges = np.linspace(e_start, e_stop, n_edges)

        source_label = "ALPHANSO Am-Be spectrum"

        if verbose:
            print(f"Total (alpha, n) yield: {an_yield:.3e} neutrons/s/g")

    except Exception as e:
        energy_edges, normalized_spectrum = ambe_spectrum_fallback_histogram()
        an_yield = np.nan
        source_label = f"Fallback Am-Be spectrum ({e})"

        if verbose:
            print(source_label)

    energy_edges = np.asarray(energy_edges, dtype=float)
    normalized_spectrum = np.asarray(normalized_spectrum, dtype=float)

    if verbose:
        print("energy_edges length:", len(energy_edges))
        print("spectrum length:", len(normalized_spectrum))

    if len(normalized_spectrum) != len(energy_edges) - 1:
        raise ValueError(
            "Spectrum length must be len(energy_edges)-1. "
            f"Got spectrum={len(normalized_spectrum)}, "
            f"edges={len(energy_edges)}"
        )

    normalized_spectrum = np.clip(normalized_spectrum, 0.0, None)

    if normalized_spectrum.sum() <= 0.0:
        raise ValueError("Spectrum contains no positive values.")

    normalized_spectrum /= normalized_spectrum.sum()

    # ALPHANSO gives bin edges in MeV.
    # OpenMC needs source energies in eV.
    energy_mid_mev = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    energy_mid_eV = energy_mid_mev * 1.0e6

    energy_prob = normalized_spectrum.copy()
    energy_prob /= energy_prob.sum()

    energy_dist = openmc.stats.Discrete(
        energy_mid_eV.tolist(),
        energy_prob.tolist()
    )

    if make_plot:
        plt.figure(figsize=(8, 5), dpi=150)
        plt.stairs(
            normalized_spectrum,
            energy_edges,
            fill=True,
            alpha=0.35,
            label=source_label
        )
        plt.title(r"$^{241}$Am-$^{9}$Be Neutron Emission Spectrum")
        plt.xlabel("Neutron energy [MeV]")
        plt.ylabel("Probability per bin")
        plt.grid(alpha=0.4)
        plt.xlim(0, 12)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if verbose:
        print(source_label)
        print("Spectrum bins:", len(normalized_spectrum))
        print("OpenMC energy points:", len(energy_mid_eV))
        print(
            f"Energy range: {energy_mid_mev.min():.2f} "
            f"to {energy_mid_mev.max():.2f} MeV"
        )
        print("Sum of probabilities:", energy_prob.sum())

    return energy_dist


# ============================================================
# 2) Base geometry and materials
# ============================================================

def build_base_geometry_and_materials(
    verbose=False,
    include_cabinet=True,
    include_lead=True,
):
    """
    Build the full base geometry and materials for the Am-Be paraffin setup.

    This includes:
      - Am-Be source capsule
      - paraffin moderator
      - plastic cylinder
      - hollow sample tubes
      - steel rods
      - steel cabinet
      - L-shaped lead shielding bricks
      - outside air region

    Parameters
    ----------
    verbose : bool
        If True, print geometry and mass summaries.

    include_cabinet : bool
        If True, include the steel cabinet wall and the cabinet inner air.
        If False, remove the cabinet wall and fill the surrounding world
        with air around the internal source/moderator geometry.

    include_lead : bool
        If True, include the L-shaped lead shielding bricks.
        If False, remove the lead bricks from the geometry.

    Returns
    -------
    base : dict
        Dictionary containing all important OpenMC objects, cells,
        materials, constants, geometry parameters, and energy_dist.
    """

    energy_dist = build_ambe_energy_distribution(
        make_plot=False,
        verbose=False
    )

    # ------------------------------------------------------------
    # Paraffin block around the AmBe source
    # ------------------------------------------------------------

    PARAFFIN_RADIUS = 29.0 / 2.0
    PARAFFIN_HEIGHT = 32.0

    PARAFFIN_ZMIN = -PARAFFIN_HEIGHT / 2.0
    PARAFFIN_ZMAX = PARAFFIN_HEIGHT / 2.0

    # ------------------------------------------------------------
    # Cabinet / safe geometry
    # ------------------------------------------------------------

    CABINET_X = 85.0
    CABINET_Y = 67.0
    CABINET_Z = 96.0

    CABINET_WALL_Y = 13.0
    CABINET_WALL_X = 15.0
    CABINET_BOTTOM = 25.0
    CABINET_TOP = 12.0

    CABINET_XMIN = -CABINET_X / 2.0
    CABINET_XMAX = CABINET_X / 2.0
    CABINET_YMIN = -CABINET_Y / 2.0
    CABINET_YMAX = CABINET_Y / 2.0

    CABINET_ZMIN = PARAFFIN_ZMIN - CABINET_BOTTOM
    CABINET_ZMAX = CABINET_ZMIN + CABINET_Z
    CABINET_ZCENTER = 0.5 * (CABINET_ZMIN + CABINET_ZMAX)

    CABINET_INNER_XMIN = CABINET_XMIN + CABINET_WALL_X
    CABINET_INNER_XMAX = CABINET_XMAX - CABINET_WALL_X
    CABINET_INNER_YMIN = CABINET_YMIN + CABINET_WALL_Y
    CABINET_INNER_YMAX = CABINET_YMAX - CABINET_WALL_Y
    CABINET_INNER_ZMIN = CABINET_ZMIN + CABINET_BOTTOM
    CABINET_INNER_ZMAX = CABINET_ZMAX - CABINET_TOP

    # ------------------------------------------------------------
    # Inner plastic cylinder / irradiation region
    # ------------------------------------------------------------

    INNER_REGION_RADIUS = 9.0 / 2.0

    PLASTIC_THICKNESS = 0.10
    PLASTIC_OUTER_RADIUS = INNER_REGION_RADIUS
    PLASTIC_INNER_RADIUS = INNER_REGION_RADIUS - PLASTIC_THICKNESS

    # ------------------------------------------------------------
    # Modeled Am-Be source capsule
    #
    # The physical source is the documented AMN 24, serial AMN 5000/1365.
    # These transport dimensions remain a geometric approximation and should
    # not be presented as factory-certified AMN 24 dimensions.
    # ------------------------------------------------------------

    SOURCE_OUTER_RADIUS = 1.91 / 2.0
    SOURCE_HEIGHT = 4.86

    SOURCE_WALL_SINGLE = 0.14
    SOURCE_WALL_TOTAL = 2.0 * SOURCE_WALL_SINGLE

    SOURCE_ACTIVE_RADIUS = SOURCE_OUTER_RADIUS - SOURCE_WALL_TOTAL
    SOURCE_ACTIVE_HEIGHT = SOURCE_HEIGHT - 2.0 * SOURCE_WALL_TOTAL

    SOURCE_ZMIN = -SOURCE_HEIGHT / 2.0
    SOURCE_ZMAX = SOURCE_HEIGHT / 2.0

    SOURCE_ACTIVE_ZMIN = -SOURCE_ACTIVE_HEIGHT / 2.0
    SOURCE_ACTIVE_ZMAX = SOURCE_ACTIVE_HEIGHT / 2.0

    # ------------------------------------------------------------
    # Hollow plastic sample tubes
    # ------------------------------------------------------------

    SAMPLE_TUBE_OUTER_RADIUS = 1.0 / 2.0
    SAMPLE_TUBE_WALL_THICKNESS = 0.10
    SAMPLE_TUBE_INNER_RADIUS = (
        SAMPLE_TUBE_OUTER_RADIUS - SAMPLE_TUBE_WALL_THICKNESS
    )

    SAMPLE_TUBE_ZMIN = 0.0
    SAMPLE_TUBE_ZMAX = PARAFFIN_ZMAX
    SAMPLE_TUBE_HEIGHT = SAMPLE_TUBE_ZMAX - SAMPLE_TUBE_ZMIN

    SAMPLE1_X, SAMPLE1_Y = -1.5, 1.3
    SAMPLE2_X, SAMPLE2_Y = 1.5, 1.3

    # ------------------------------------------------------------
    # Steel rods
    # ------------------------------------------------------------

    ROD_RADIUS = 0.5 / 2.0

    ROD1_X, ROD1_Y = 0.0, 10.0
    ROD2_X, ROD2_Y = 0.0, -10.0

    # ------------------------------------------------------------
    # Lead shielding bricks, L-shape with 90° bend
    # ------------------------------------------------------------

    LEAD_BRICK_LENGTH = 20.0
    LEAD_BRICK_DEPTH = 5.0
    LEAD_BRICK_HEIGHT = 10.0

    LEAD_NZ = 3
    LEAD_TOP_N = 2
    LEAD_SIDE_N = 2

    LEAD_STACK_ZMIN = CABINET_INNER_ZMIN
    LEAD_STACK_ZMAX = (LEAD_STACK_ZMIN + LEAD_NZ * LEAD_BRICK_HEIGHT)

    # Upper horizontal leg
    LEAD_TOP_YMIN = 15.5
    LEAD_TOP_YMAX = LEAD_TOP_YMIN + LEAD_BRICK_DEPTH

    # Right vertical leg
    LEAD_RIGHT_XMIN = 15.5
    LEAD_RIGHT_XMAX = LEAD_RIGHT_XMIN + LEAD_BRICK_DEPTH

    # Make the two complete 40-cm legs meet without overlapping
    LEAD_TOP_XMAX = LEAD_RIGHT_XMIN
    LEAD_TOP_XMIN = (LEAD_TOP_XMAX - LEAD_TOP_N * LEAD_BRICK_LENGTH)

    LEAD_RIGHT_YMAX = LEAD_TOP_YMAX
    LEAD_RIGHT_YMIN = (LEAD_RIGHT_YMAX - LEAD_SIDE_N * LEAD_BRICK_LENGTH)

    # ------------------------------------------------------------
    # World boundary
    # ------------------------------------------------------------

    WORLD_HALF_X = 700.0
    WORLD_HALF_Y = 700.0
    WORLD_HALF_Z = 500.0

    if verbose:
        print("Geometry summary:")
        print(f"  Cabinet outer dimensions      = {CABINET_X:.1f} x {CABINET_Y:.1f} x {CABINET_Z:.1f} cm")
        print(f"  Cabinet x-wall thickness      = {CABINET_WALL_X:.1f} cm")
        print(f"  Cabinet y-wall thickness      = {CABINET_WALL_Y:.1f} cm")
        print(f"  Cabinet bottom thickness      = {CABINET_BOTTOM:.1f} cm")
        print(f"  Cabinet top thickness         = {CABINET_TOP:.1f} cm")
        print(f"  Cabinet z range               = {CABINET_ZMIN:.1f} to {CABINET_ZMAX:.1f} cm")
        print(f"  Cabinet inner x width         = {CABINET_INNER_XMAX - CABINET_INNER_XMIN:.1f} cm")
        print(f"  Cabinet inner y width         = {CABINET_INNER_YMAX - CABINET_INNER_YMIN:.1f} cm")
        print(f"  Cabinet inner z height        = {CABINET_INNER_ZMAX - CABINET_INNER_ZMIN:.1f} cm")
        print(f"  Cabinet inner z range         = {CABINET_INNER_ZMIN:.1f} to {CABINET_INNER_ZMAX:.1f} cm")
        print(f"  Paraffin radius               = {PARAFFIN_RADIUS:.3f} cm")
        print(f"  Paraffin height               = {PARAFFIN_HEIGHT:.3f} cm")
        print(f"  Paraffin z range              = {PARAFFIN_ZMIN:.3f} to {PARAFFIN_ZMAX:.3f} cm")
        print(f"  Plastic outer radius          = {PLASTIC_OUTER_RADIUS:.3f} cm")
        print(f"  Plastic inner radius          = {PLASTIC_INNER_RADIUS:.3f} cm")
        print(f"  Source outer radius           = {SOURCE_OUTER_RADIUS:.3f} cm")
        print(f"  Source active radius          = {SOURCE_ACTIVE_RADIUS:.3f} cm")
        print(f"  Source active height          = {SOURCE_ACTIVE_HEIGHT:.3f} cm")

    # ------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------

    air = openmc.Material(name="Air")
    air.set_density("g/cm3", 0.001205)
    air.add_element("N", 0.78, percent_type="ao")
    air.add_element("O", 0.21, percent_type="ao")
    air.add_element("Ar", 0.01, percent_type="ao")

    paraffin = openmc.Material(name="Paraffin CH2")
    paraffin.set_density("g/cm3", 0.93)
    paraffin.add_element("C", 1.0, percent_type="ao")
    paraffin.add_element("H", 2.0, percent_type="ao")

    try:
        paraffin.add_s_alpha_beta("c_H_in_CH2")
    except Exception as e:
        if verbose:
            print("Note: thermal scattering c_H_in_CH2 not added for paraffin:", e)

    plastic = openmc.Material(name="Plastic CH2")
    plastic.set_density("g/cm3", 0.95)
    plastic.add_element("C", 1.0, percent_type="ao")
    plastic.add_element("H", 2.0, percent_type="ao")

    try:
        plastic.add_s_alpha_beta("c_H_in_CH2")
    except Exception as e:
        if verbose:
            print("Note: thermal scattering c_H_in_CH2 not added for plastic:", e)

    cabinet_wall = openmc.Material(name="Cabinet wall material")
    cabinet_wall.set_density("g/cm3", 7.85)
    cabinet_wall.add_element("Fe", 0.98, percent_type="wo")
    cabinet_wall.add_element("C", 0.02, percent_type="wo")

    steel = openmc.Material(name="Stainless steel")
    steel.set_density("g/cm3", 8.0)
    steel.add_element("Fe", 0.70, percent_type="wo")
    steel.add_element("Cr", 0.19, percent_type="wo")
    steel.add_element("Ni", 0.10, percent_type="wo")
    steel.add_element("Mn", 0.01, percent_type="wo")

    lead = openmc.Material(name="Lead shielding bricks")
    lead.set_density("g/cm3", 11.34)
    lead.add_element("Pb", 1.0, percent_type="wo")

    ambe_fill = openmc.Material(name="AmBe fill transport approximation")
    ambe_fill.set_density("g/cm3", 1.85)
    ambe_fill.add_element("Be", 1.0, percent_type="ao")

    materials = openmc.Materials([
        air,
        paraffin,
        plastic,
        cabinet_wall,
        steel,
        lead,
        ambe_fill,
    ])

    # ------------------------------------------------------------
    # Surfaces
    # ------------------------------------------------------------

    x_w_min = openmc.XPlane(x0=-WORLD_HALF_X, boundary_type="vacuum")
    x_w_max = openmc.XPlane(x0=WORLD_HALF_X, boundary_type="vacuum")
    y_w_min = openmc.YPlane(y0=-WORLD_HALF_Y, boundary_type="vacuum")
    y_w_max = openmc.YPlane(y0=WORLD_HALF_Y, boundary_type="vacuum")
    z_w_min = openmc.ZPlane(z0=-WORLD_HALF_Z, boundary_type="vacuum")
    z_w_max = openmc.ZPlane(z0=WORLD_HALF_Z, boundary_type="vacuum")

    world_region = (
        +x_w_min & -x_w_max &
        +y_w_min & -y_w_max &
        +z_w_min & -z_w_max
    )

    cab_x_min = openmc.XPlane(x0=CABINET_XMIN)
    cab_x_max = openmc.XPlane(x0=CABINET_XMAX)
    cab_y_min = openmc.YPlane(y0=CABINET_YMIN)
    cab_y_max = openmc.YPlane(y0=CABINET_YMAX)
    cab_z_min = openmc.ZPlane(z0=CABINET_ZMIN)
    cab_z_max = openmc.ZPlane(z0=CABINET_ZMAX)

    cabinet_outer_region = (
        +cab_x_min & -cab_x_max &
        +cab_y_min & -cab_y_max &
        +cab_z_min & -cab_z_max
    )

    cav_x_min = openmc.XPlane(x0=CABINET_INNER_XMIN)
    cav_x_max = openmc.XPlane(x0=CABINET_INNER_XMAX)
    cav_y_min = openmc.YPlane(y0=CABINET_INNER_YMIN)
    cav_y_max = openmc.YPlane(y0=CABINET_INNER_YMAX)
    cav_z_min = openmc.ZPlane(z0=CABINET_INNER_ZMIN)
    cav_z_max = openmc.ZPlane(z0=CABINET_INNER_ZMAX)

    cabinet_inner_region = (
        +cav_x_min & -cav_x_max &
        +cav_y_min & -cav_y_max &
        +cav_z_min & -cav_z_max
    )

    cabinet_wall_region = cabinet_outer_region & ~cabinet_inner_region

    paraffin_cyl = openmc.ZCylinder(r=PARAFFIN_RADIUS)
    paraffin_bot = openmc.ZPlane(z0=PARAFFIN_ZMIN)
    paraffin_top = openmc.ZPlane(z0=PARAFFIN_ZMAX)

    paraffin_total_region = (
        -paraffin_cyl &
        +paraffin_bot &
        -paraffin_top
    )

    plastic_outer_cyl = openmc.ZCylinder(r=PLASTIC_OUTER_RADIUS)
    plastic_inner_cyl = openmc.ZCylinder(r=PLASTIC_INNER_RADIUS)

    plastic_region = (
        -plastic_outer_cyl &
        +plastic_inner_cyl &
        +paraffin_bot &
        -paraffin_top
    )

    source_outer_cyl = openmc.ZCylinder(r=SOURCE_OUTER_RADIUS)
    source_outer_bot = openmc.ZPlane(z0=SOURCE_ZMIN)
    source_outer_top = openmc.ZPlane(z0=SOURCE_ZMAX)

    source_outer_region = (
        -source_outer_cyl &
        +source_outer_bot &
        -source_outer_top
    )

    source_active_cyl = openmc.ZCylinder(r=SOURCE_ACTIVE_RADIUS)
    source_active_bot = openmc.ZPlane(z0=SOURCE_ACTIVE_ZMIN)
    source_active_top = openmc.ZPlane(z0=SOURCE_ACTIVE_ZMAX)

    source_active_region = (
        -source_active_cyl &
        +source_active_bot &
        -source_active_top
    )

    source_shell_region = source_outer_region & ~source_active_region

    sample_tube_bot = openmc.ZPlane(z0=SAMPLE_TUBE_ZMIN)
    sample_tube_top = openmc.ZPlane(z0=SAMPLE_TUBE_ZMAX)

    sample1_outer_cyl = openmc.ZCylinder(
        x0=SAMPLE1_X,
        y0=SAMPLE1_Y,
        r=SAMPLE_TUBE_OUTER_RADIUS
    )

    sample2_outer_cyl = openmc.ZCylinder(
        x0=SAMPLE2_X,
        y0=SAMPLE2_Y,
        r=SAMPLE_TUBE_OUTER_RADIUS
    )

    sample1_inner_cyl = openmc.ZCylinder(
        x0=SAMPLE1_X,
        y0=SAMPLE1_Y,
        r=SAMPLE_TUBE_INNER_RADIUS
    )

    sample2_inner_cyl = openmc.ZCylinder(
        x0=SAMPLE2_X,
        y0=SAMPLE2_Y,
        r=SAMPLE_TUBE_INNER_RADIUS
    )

    sample1_total_region = (
        -sample1_outer_cyl &
        +sample_tube_bot &
        -sample_tube_top
    )

    sample2_total_region = (
        -sample2_outer_cyl &
        +sample_tube_bot &
        -sample_tube_top
    )

    sample1_region = (
        -sample1_inner_cyl &
        +sample_tube_bot &
        -sample_tube_top
    )

    sample2_region = (
        -sample2_inner_cyl &
        +sample_tube_bot &
        -sample_tube_top
    )

    sample1_plastic_region = (
        -sample1_outer_cyl &
        +sample1_inner_cyl &
        +sample_tube_bot &
        -sample_tube_top
    )

    sample2_plastic_region = (
        -sample2_outer_cyl &
        +sample2_inner_cyl &
        +sample_tube_bot &
        -sample_tube_top
    )

    ROD_EXTENSION_ABOVE_PARAFFIN = 15.0

    ROD_ZMIN = PARAFFIN_ZMIN
    ROD_ZMAX = PARAFFIN_ZMAX + ROD_EXTENSION_ABOVE_PARAFFIN
    ROD_HEIGHT = ROD_ZMAX - ROD_ZMIN

    rod_zmin_plane = openmc.ZPlane(z0=ROD_ZMIN)
    rod_zmax_plane = openmc.ZPlane(z0=ROD_ZMAX)

    rod1_cyl = openmc.ZCylinder(
        x0=ROD1_X,
        y0=ROD1_Y,
        r=ROD_RADIUS
    )

    rod2_cyl = openmc.ZCylinder(
        x0=ROD2_X,
        y0=ROD2_Y,
        r=ROD_RADIUS
    )

    rod1_region = (
        -rod1_cyl &
        +rod_zmin_plane &
        -rod_zmax_plane
    )

    rod2_region = (
        -rod2_cyl &
        +rod_zmin_plane &
        -rod_zmax_plane
    )

    def make_box_region(xmin, xmax, ymin, ymax, zmin, zmax):
        x_min = openmc.XPlane(x0=xmin)
        x_max = openmc.XPlane(x0=xmax)
        y_min = openmc.YPlane(y0=ymin)
        y_max = openmc.YPlane(y0=ymax)
        z_min = openmc.ZPlane(z0=zmin)
        z_max = openmc.ZPlane(z0=zmax)

        return (
            +x_min & -x_max &
            +y_min & -y_max &
            +z_min & -z_max
        )

    LEAD_BRICKS = []
    lead_brick_regions = []
    lead_brick_cells = []

    brick_id = 1

    # Top horizontal leg: 2 bricks along x, 3 layers in z
    for iz in range(LEAD_NZ):
        zmin = LEAD_STACK_ZMIN + iz * LEAD_BRICK_HEIGHT
        zmax = zmin + LEAD_BRICK_HEIGHT

        for ix in range(LEAD_TOP_N):
            xmin = LEAD_TOP_XMIN + ix * LEAD_BRICK_LENGTH
            xmax = xmin + LEAD_BRICK_LENGTH
            ymin = LEAD_TOP_YMIN
            ymax = LEAD_TOP_YMAX

            brick = {
                "name": f"Lead brick {brick_id}",
                "part": "top",
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "zmin": zmin,
                "zmax": zmax,
            }

            LEAD_BRICKS.append(brick)

            brick_region = make_box_region(
                xmin, xmax,
                ymin, ymax,
                zmin, zmax
            )

            lead_brick_regions.append(brick_region)

            if include_lead:
                lead_cell = openmc.Cell(
                    name=brick["name"],
                    fill=lead,
                    region=brick_region
                )

                lead_brick_cells.append(lead_cell)

            brick_id += 1

    # Right vertical leg: 2 bricks along y, 3 layers in z
    for iz in range(LEAD_NZ):
        zmin = LEAD_STACK_ZMIN + iz * LEAD_BRICK_HEIGHT
        zmax = zmin + LEAD_BRICK_HEIGHT

        for iy in range(LEAD_SIDE_N):
            xmin = LEAD_RIGHT_XMIN
            xmax = LEAD_RIGHT_XMAX
            ymin = LEAD_RIGHT_YMIN + iy * LEAD_BRICK_LENGTH
            ymax = ymin + LEAD_BRICK_LENGTH

            brick = {
                "name": f"Lead brick {brick_id}",
                "part": "right",
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "zmin": zmin,
                "zmax": zmax,
            }

            LEAD_BRICKS.append(brick)

            brick_region = make_box_region(
                xmin, xmax,
                ymin, ymax,
                zmin, zmax
            )

            lead_brick_regions.append(brick_region)

            if include_lead:
                lead_cell = openmc.Cell(
                    name=brick["name"],
                    fill=lead,
                    region=brick_region
                )

                lead_brick_cells.append(lead_cell)

            brick_id += 1

    inner_paraffin_total_region = (
        -plastic_inner_cyl &
        +paraffin_bot &
        -paraffin_top
    )

    inner_paraffin_region = (
        inner_paraffin_total_region &
        ~source_outer_region &
        ~sample1_total_region &
        ~sample2_total_region
    )

    outer_paraffin_region = (
        paraffin_total_region &
        ~inner_paraffin_total_region &
        ~plastic_region &
        ~rod1_region &
        ~rod2_region
    )

    if include_cabinet:
        cabinet_air_region = (
            cabinet_inner_region &
            ~paraffin_total_region &
            ~rod1_region &
            ~rod2_region
        )

        if include_lead:
            for lead_region in lead_brick_regions:
                cabinet_air_region = cabinet_air_region & ~lead_region

        outside_air_region = world_region & ~cabinet_outer_region

    else:
        cabinet_air_region = None
        cabinet_wall_region = None

        outside_air_region = (
            world_region &
            ~paraffin_total_region &
            ~rod1_region &
            ~rod2_region
        )

        if include_lead:
            for lead_region in lead_brick_regions:
                outside_air_region = outside_air_region & ~lead_region

    # ------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------

    if include_cabinet:
        cabinet_wall_cell = openmc.Cell(
            name="Cabinet wall",
            fill=cabinet_wall,
            region=cabinet_wall_region
        )

        cabinet_air_cell = openmc.Cell(
            name="Cabinet inner air",
            fill=air,
            region=cabinet_air_region
        )

    else:
        cabinet_wall_cell = None
        cabinet_air_cell = None

    inner_paraffin_cell = openmc.Cell(
        name="Inner paraffin block around source",
        fill=paraffin,
        region=inner_paraffin_region
    )

    outer_paraffin_cell = openmc.Cell(
        name="Outer paraffin moderator",
        fill=paraffin,
        region=outer_paraffin_region
    )

    plastic_cell = openmc.Cell(
        name="Plastic cylinder",
        fill=plastic,
        region=plastic_region
    )

    source_fill_cell = openmc.Cell(
        name="AmBe source fill",
        fill=ambe_fill,
        region=source_active_region
    )

    source_shell_cell = openmc.Cell(
        name="Source stainless steel double capsule",
        fill=steel,
        region=source_shell_region
    )

    sample1_cell = openmc.Cell(
        name="Sample tube 1 inner air",
        fill=air,
        region=sample1_region
    )

    sample2_cell = openmc.Cell(
        name="Sample tube 2 inner air",
        fill=air,
        region=sample2_region
    )

    sample1_plastic_cell = openmc.Cell(
        name="Sample tube 1 plastic wall",
        fill=plastic,
        region=sample1_plastic_region
    )

    sample2_plastic_cell = openmc.Cell(
        name="Sample tube 2 plastic wall",
        fill=plastic,
        region=sample2_plastic_region
    )

    rod1_cell = openmc.Cell(
        name="Metal rod 1",
        fill=steel,
        region=rod1_region
    )

    rod2_cell = openmc.Cell(
        name="Metal rod 2",
        fill=steel,
        region=rod2_region
    )

    outside_air_cell = openmc.Cell(
        name="Outside air",
        fill=air,
        region=outside_air_region
    )

    root_cells = [
        inner_paraffin_cell,
        outer_paraffin_cell,
        plastic_cell,
        source_fill_cell,
        source_shell_cell,
        sample1_cell,
        sample2_cell,
        sample1_plastic_cell,
        sample2_plastic_cell,
        rod1_cell,
        rod2_cell,
        outside_air_cell,
    ]

    if include_cabinet:
        root_cells.insert(0, cabinet_air_cell)
        root_cells.insert(0, cabinet_wall_cell)

    if include_lead:
        insert_index = len(root_cells) - 1
        root_cells[insert_index:insert_index] = lead_brick_cells

    root = openmc.Universe(cells=root_cells)

    geometry = openmc.Geometry(root)

    # ------------------------------------------------------------
    # Approximate mass checks
    # ------------------------------------------------------------

    paraffin_total_volume_cm3 = np.pi * PARAFFIN_RADIUS**2 * PARAFFIN_HEIGHT
    source_active_volume_cm3 = np.pi * SOURCE_ACTIVE_RADIUS**2 * SOURCE_ACTIVE_HEIGHT

    cabinet_outer_volume_cm3 = CABINET_X * CABINET_Y * CABINET_Z
    cabinet_inner_volume_cm3 = (
        (CABINET_INNER_XMAX - CABINET_INNER_XMIN) *
        (CABINET_INNER_YMAX - CABINET_INNER_YMIN) *
        (CABINET_INNER_ZMAX - CABINET_INNER_ZMIN)
    )
    cabinet_wall_volume_cm3 = cabinet_outer_volume_cm3 - cabinet_inner_volume_cm3

    approx_paraffin_mass_kg = paraffin_total_volume_cm3 * 0.93 / 1000.0
    approx_active_source_mass_g = source_active_volume_cm3 * 1.85
    approx_cabinet_wall_mass_kg = cabinet_wall_volume_cm3 * 7.85 / 1000.0

    lead_total_volume_cm3 = (
        len(lead_brick_cells) *
        LEAD_BRICK_LENGTH *
        LEAD_BRICK_DEPTH *
        LEAD_BRICK_HEIGHT
    )

    approx_lead_mass_kg = lead_total_volume_cm3 * 11.34 / 1000.0

    if verbose:
        print("\nMaterials and geometry built.")
        print("Defined cells:")
        for cell in root.cells.values():
            print(f"  {cell.id}: {cell.name}")

        print()
        print("Sample tubes:")
        print(f"  Sample tube outer radius       = {SAMPLE_TUBE_OUTER_RADIUS:.3f} cm")
        print(f"  Sample tube inner radius       = {SAMPLE_TUBE_INNER_RADIUS:.3f} cm")
        print(f"  Sample tube wall thickness     = {SAMPLE_TUBE_WALL_THICKNESS:.3f} cm")
        print(f"  Sample tube z range            = {SAMPLE_TUBE_ZMIN:.3f} to {SAMPLE_TUBE_ZMAX:.3f} cm")
        print(f"  Sample tube height             = {SAMPLE_TUBE_HEIGHT:.3f} cm")

        print("\nApproximate mass checks:")
        print(f"  Total paraffin volume:       {paraffin_total_volume_cm3:.1f} cm^3")
        print(f"  Approx. paraffin mass:       {approx_paraffin_mass_kg:.2f} kg")
        print(f"  Active source volume:        {source_active_volume_cm3:.3f} cm^3")
        print(f"  Approx. active fill mass:    {approx_active_source_mass_g:.3f} g")
        print(f"  Cabinet wall volume:         {cabinet_wall_volume_cm3:.1f} cm^3")
        print(f"  Approx. cabinet wall mass:   {approx_cabinet_wall_mass_kg:.2f} kg")

        print()
        print("Lead shielding:")
        print(f"  Lead bricks:                 {len(lead_brick_cells)}")
        print(f"  Single lead brick:           {LEAD_BRICK_LENGTH:.1f} x {LEAD_BRICK_DEPTH:.1f} x {LEAD_BRICK_HEIGHT:.1f} cm")
        print(f"  Top leg bricks/layer:        {LEAD_TOP_N}")
        print(f"  Right leg bricks/layer:      {LEAD_SIDE_N}")
        print(f"  Vertical layers:             {LEAD_NZ}")
        print(f"  Top leg x range:             {LEAD_TOP_XMIN:.1f} to {LEAD_TOP_XMAX:.1f} cm")
        print(f"  Top leg y range:             {LEAD_TOP_YMIN:.1f} to {LEAD_TOP_YMAX:.1f} cm")
        print(f"  Right leg x range:           {LEAD_RIGHT_XMIN:.1f} to {LEAD_RIGHT_XMAX:.1f} cm")
        print(f"  Right leg y range:           {LEAD_RIGHT_YMIN:.1f} to {LEAD_RIGHT_YMAX:.1f} cm")
        print(f"  Lead z range:                {LEAD_STACK_ZMIN:.1f} to {LEAD_STACK_ZMAX:.1f} cm")
        print(f"  Lead total volume:           {lead_total_volume_cm3:.1f} cm^3")
        print(f"  Approx. lead mass:           {approx_lead_mass_kg:.2f} kg")

    base = locals().copy()

    # Avoid exporting helper function in base dictionary.
    base.pop("make_box_region", None)

    return base


# ============================================================
# 3) Complete base model
# ============================================================

def build_base_model(
    export_xml=False,
    output_dir=None,
    verbose=False,
    include_cabinet=True,
    include_lead=True,
    source_evaluation_date=None,
):
    """
    Build the complete shared base OpenMC model.

    Parameters
    ----------
    export_xml : bool
        If True, export OpenMC XML files.

    output_dir : str or pathlib.Path or None
        Directory where XML files should be exported.
        If None, XML files are written to the current working directory.

    verbose : bool
        If True, print geometry summaries.

    source_evaluation_date : str, datetime.date, datetime.datetime, or None
        Experimental date used to decay-correct the 1968 factory neutron
        emission. If None, retain and flag the reference-date value.

    Returns
    -------
    base : dict
        Dictionary containing the OpenMC model, geometry, materials,
        energy distribution, cells, and constants.
    """

    base = build_base_geometry_and_materials(
        verbose=verbose,
        include_cabinet=include_cabinet,
        include_lead=include_lead,
    )
    base.update(get_source_record(source_evaluation_date))

    model = openmc.Model(
        geometry=base["geometry"],
        materials=base["materials"]
    )

    base["model"] = model

    if export_xml:
        if output_dir is None:
            model.export_to_xml()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model.export_to_xml(directory=output_dir)

    return base


# ============================================================
# 5) Closed-box surface-current leakage helpers
# ============================================================

SOURCE_GAMMA_EMISSION = None  # photons/s; set in notebook if an absolute rate is used


def build_gamma_source(gamma_energy_dist, source_strength=1.0):
    """
    Build an isotropic fixed-source photon source at the Am-Be capsule centre.

    The source is normalized to source_strength. Physical scaling is applied
    in post-processing, analogous to SOURCE_NEUTRON_EMISSION.
    """

    space_dist = openmc.stats.Point((0.0, 0.0, 0.0))
    angle_dist = openmc.stats.Isotropic()

    gamma_source = openmc.IndependentSource(
        space=space_dist,
        angle=angle_dist,
        energy=gamma_energy_dist,
        particle="photon",
    )
    gamma_source.strength = source_strength

    return gamma_source


def build_gamma_settings(
    gamma_energy_dist,
    batches=120,
    particles=50_000,
    inactive=0,
    source_strength=1.0,
):
    """Build OpenMC fixed-source settings for a photon-source run."""

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = build_gamma_source(
        gamma_energy_dist=gamma_energy_dist,
        source_strength=source_strength,
    )
    settings.batches = batches
    settings.particles = particles
    settings.inactive = inactive

    return settings


def define_closed_leakage_box(
    base,
    margin_cm=15.0,
    size_x_cm=600.0,
    size_y_cm=600.0,
    size_z_cm=250.0,
):
    """
    Define a rectangular closed leakage box around the cabinet reference geometry.

    Physical placement used for Notebook 1.5:
      - x_min wall is margin_cm from cabinet x_min,
      - y_max wall is margin_cm from cabinet y_max,
      - z_min wall is margin_cm below cabinet z_min.

    This places the full cabinet/source geometry close to one corner of the
    closed box and close to the floor. The same reference box is used for both
    the full-geometry and no-cabinet/no-lead scenarios.
    """

    box_xmin = base["CABINET_XMIN"] - margin_cm
    box_xmax = box_xmin + size_x_cm

    box_ymax = base["CABINET_YMAX"] + margin_cm
    box_ymin = box_ymax - size_y_cm

    box_zmin = base["CABINET_ZMIN"] - margin_cm
    box_zmax = box_zmin + size_z_cm

    contains_cabinet_reference = (
        box_xmin < base["CABINET_XMIN"]
        and box_xmax > base["CABINET_XMAX"]
        and box_ymin < base["CABINET_YMIN"]
        and box_ymax > base["CABINET_YMAX"]
        and box_zmin < base["CABINET_ZMIN"]
        and box_zmax > base["CABINET_ZMAX"]
    )

    if not contains_cabinet_reference:
        raise ValueError(
            "The closed leakage box does not contain the cabinet reference geometry."
        )

    surfaces = {
        "x_min": openmc.XPlane(x0=box_xmin),
        "x_max": openmc.XPlane(x0=box_xmax),
        "y_min": openmc.YPlane(y0=box_ymin),
        "y_max": openmc.YPlane(y0=box_ymax),
        "z_min": openmc.ZPlane(z0=box_zmin),
        "z_max": openmc.ZPlane(z0=box_zmax),
    }

    box_region = (
        +surfaces["x_min"] & -surfaces["x_max"] &
        +surfaces["y_min"] & -surfaces["y_max"] &
        +surfaces["z_min"] & -surfaces["z_max"]
    )

    side_info = {
        "x_min": {
            "normal": (-1.0, 0.0, 0.0),
            "outward_sign": -1.0,
            "area_cm2": size_y_cm * size_z_cm,
        },
        "x_max": {
            "normal": (+1.0, 0.0, 0.0),
            "outward_sign": +1.0,
            "area_cm2": size_y_cm * size_z_cm,
        },
        "y_min": {
            "normal": (0.0, -1.0, 0.0),
            "outward_sign": -1.0,
            "area_cm2": size_x_cm * size_z_cm,
        },
        "y_max": {
            "normal": (0.0, +1.0, 0.0),
            "outward_sign": +1.0,
            "area_cm2": size_x_cm * size_z_cm,
        },
        "z_min": {
            "normal": (0.0, 0.0, -1.0),
            "outward_sign": -1.0,
            "area_cm2": size_x_cm * size_y_cm,
        },
        "z_max": {
            "normal": (0.0, 0.0, +1.0),
            "outward_sign": +1.0,
            "area_cm2": size_x_cm * size_y_cm,
        },
    }

    return {
        "margin_cm": margin_cm,
        "size_x_cm": size_x_cm,
        "size_y_cm": size_y_cm,
        "size_z_cm": size_z_cm,
        "xmin": box_xmin,
        "xmax": box_xmax,
        "ymin": box_ymin,
        "ymax": box_ymax,
        "zmin": box_zmin,
        "zmax": box_zmax,
        "surfaces": surfaces,
        "box_region": box_region,
        "side_info": side_info,
    }


def build_surface_current_tallies(
    leakage_box,
    particle=None,
    tally_prefix="surface_current",
):
    """
    Build one OpenMC surface-current tally per side of a closed box.

    Parameters
    ----------
    leakage_box : dict
        Dictionary returned by define_closed_leakage_box(...).

    particle : str or None
        If given, apply an OpenMC ParticleFilter. Typical values are
        "neutron" or "photon". If None, no particle filter is used.

    tally_prefix : str
        Prefix used in tally names.
    """

    tallies = []
    tally_by_side = {}

    particle_filter = None
    if particle is not None:
        particle_filter = openmc.ParticleFilter([particle])

    for side_name, surface in leakage_box["surfaces"].items():
        tally = openmc.Tally(name=f"{tally_prefix}_{side_name}")
        filters = [openmc.SurfaceFilter(surface)]

        if particle_filter is not None:
            filters.append(particle_filter)

        tally.filters = filters
        tally.scores = ["current"]

        tallies.append(tally)
        tally_by_side[side_name] = tally

    return {
        "tallies": openmc.Tallies(tallies),
        "tally_by_side": tally_by_side,
    }


def build_surface_current_model(
    include_cabinet=True,
    include_lead=True,
    particle="neutron",
    batches=120,
    particles=50_000,
    inactive=0,
    box_margin_cm=15.0,
    box_size_x_cm=600.0,
    box_size_y_cm=600.0,
    box_size_z_cm=250.0,
    verbose=False,
    source_evaluation_date=None,
):
    """
    Build an OpenMC model for a closed-box surface-current leakage calculation.

    This helper supports two source runs:
      - particle="neutron": Am-Be neutron source and neutron current tallies.
      - particle="photon": Am-Be gamma-line photon source and photon current tallies.

    Important OpenMC detail:
    The leakage-box surfaces must be part of the geometry for SurfaceFilter
    tallies to work. Therefore this function splits the original outside-air
    cell into:
      1. outside air inside the leakage box,
      2. air outside the leakage box but still inside the OpenMC world.

    ``source_evaluation_date`` is stored with the returned source record. It
    does not alter the per-source-particle OpenMC transport calculation.
    """

    base = build_base_geometry_and_materials(
        verbose=verbose,
        include_cabinet=include_cabinet,
        include_lead=include_lead,
    )
    base.update(get_source_record(source_evaluation_date))

    leakage_box = define_closed_leakage_box(
        base=base,
        margin_cm=box_margin_cm,
        size_x_cm=box_size_x_cm,
        size_y_cm=box_size_y_cm,
        size_z_cm=box_size_z_cm,
    )

    # ------------------------------------------------------------
    # Insert the leakage box as a real geometry boundary.
    #
    # This makes the six leakage-box planes appear in geometry.xml.
    # Without this, OpenMC cannot use them in a SurfaceFilter.
    # ------------------------------------------------------------

    box_region = leakage_box["box_region"]

    outside_air_cell = base["outside_air_cell"]
    outside_air_cell.region = outside_air_cell.region & box_region

    outside_box_air_cell = openmc.Cell(
        name="Air outside closed leakage box",
        fill=base["air"],
        region=base["world_region"] & ~box_region,
    )

    root = base["geometry"].root_universe
    root.add_cell(outside_box_air_cell)

    base["outside_box_air_cell"] = outside_box_air_cell

    # ------------------------------------------------------------
    # Source settings
    # ------------------------------------------------------------

    if particle == "neutron":
        settings = build_neutron_settings(
            energy_dist=base["energy_dist"],
            batches=batches,
            particles=particles,
            inactive=inactive,
            source_strength=1.0,
        )
        tally_particle = "neutron"

    elif particle == "photon":
        gamma_energy_dist, gamma_data = build_ambe_gamma_line_distribution(
            make_plot=False,
            verbose=verbose,
        )

        settings = build_gamma_settings(
            gamma_energy_dist=gamma_energy_dist,
            batches=batches,
            particles=particles,
            inactive=inactive,
            source_strength=1.0,
        )

        base["gamma_energy_dist"] = gamma_energy_dist
        base["gamma_data"] = gamma_data
        tally_particle = "photon"

    else:
        raise ValueError("particle must be either 'neutron' or 'photon'.")

    # ------------------------------------------------------------
    # Surface-current tallies
    # ------------------------------------------------------------

    tally_data = build_surface_current_tallies(
        leakage_box=leakage_box,
        particle=tally_particle,
        tally_prefix=f"{particle}_current",
    )

    model = openmc.Model(
        geometry=base["geometry"],
        materials=base["materials"],
        settings=settings,
        tallies=tally_data["tallies"],
    )

    model_data = {}
    model_data.update(base)
    model_data.update(tally_data)

    model_data["settings"] = settings
    model_data["model"] = model
    model_data["leakage_box"] = leakage_box
    model_data["particle"] = particle
    model_data["include_cabinet"] = include_cabinet
    model_data["include_lead"] = include_lead
    model_data["outside_air_cell_restricted_to_box"] = True

    return model_data



# ============================================================
# 4) Neutron fixed-source helpers and base tally model
# ============================================================

# Backward-compatible module-level value.  This is the documented factory
# emission at the 1968-07-05 reference date.  ``build_base_neutron_model``
# replaces it in the returned model dictionary with the decay-corrected value
# when ``source_evaluation_date`` is supplied.
SOURCE_NEUTRON_EMISSION = SOURCE_NEUTRON_EMISSION_REF


def build_neutron_source(energy_dist, source_strength=1.0):
    """Build an isotropic fixed-source neutron source at the Am-Be centre."""

    space_dist = openmc.stats.Point((0.0, 0.0, 0.0))
    angle_dist = openmc.stats.Isotropic()

    ambe_source = openmc.IndependentSource(
        space=space_dist,
        angle=angle_dist,
        energy=energy_dist,
        particle="neutron",
    )
    ambe_source.strength = source_strength

    return ambe_source


def build_neutron_settings(
    energy_dist,
    batches=120,
    particles=50_000,
    inactive=0,
    source_strength=1.0,
):
    """Build OpenMC fixed-source settings for a neutron-source run."""

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = build_neutron_source(
        energy_dist=energy_dist,
        source_strength=source_strength,
    )
    settings.batches = batches
    settings.particles = particles
    settings.inactive = inactive

    return settings


def z_from_cabinet_bottom(h_cm, cabinet_zmin=None):
    """Convert height from the outer cabinet bottom to OpenMC z-coordinate."""

    if cabinet_zmin is None:
        # Default from the current geometry: paraffin zmin = -16 cm,
        # cabinet bottom thickness = 25 cm.
        cabinet_zmin = -41.0

    return float(cabinet_zmin) + float(h_cm)


def build_neutron_energy_filter(energy_bins=None):
    """Build the energy filter used for neutron spectral tallies."""

    if energy_bins is None:
        energy_bins = np.logspace(-3, 8, 180)  # 1e-3 eV to 100 MeV
    else:
        energy_bins = np.asarray(energy_bins, dtype=float)

    energy_filter = openmc.EnergyFilter(energy_bins)
    return energy_bins, energy_filter


def make_mesh_flux_tally(
    name,
    center,
    det_size=2.0,
    spectral=False,
    energy_filter=None,
):
    """Create a 1x1x1 regular-mesh flux tally centered at `center`."""

    if spectral and energy_filter is None:
        raise ValueError("energy_filter must be supplied when spectral=True.")

    x, y, z = center
    half = det_size / 2.0

    mesh = openmc.RegularMesh()
    mesh.dimension = [1, 1, 1]
    mesh.lower_left = [x - half, y - half, z - half]
    mesh.upper_right = [x + half, y + half, z + half]

    mesh_filter = openmc.MeshFilter(mesh)

    tally = openmc.Tally(name=name)
    tally.filters = [mesh_filter, energy_filter] if spectral else [mesh_filter]
    tally.scores = ["flux"]

    return tally


def build_base_tally_points(base, det_offset=1.0):
    """
    Build the reference P1--P5 tally positions used in Notebook 1.

    The points are measurement-like reference locations just outside the cabinet.
    Heights are measured from the outer cabinet bottom and then converted to
    the OpenMC z coordinate.
    """

    cabinet_zmin = base["CABINET_ZMIN"]
    corner_offset = det_offset / np.sqrt(2.0)

    points = {
        "P1_h42": (
            0.0,
            base["CABINET_YMIN"] - det_offset,
            z_from_cabinet_bottom(42.0, cabinet_zmin),
        ),
        "P1_h85": (
            0.0,
            base["CABINET_YMIN"] - det_offset,
            z_from_cabinet_bottom(85.0, cabinet_zmin),
        ),
        "P2_h55": (
            base["CABINET_XMAX"] + corner_offset,
            base["CABINET_YMIN"] - corner_offset,
            z_from_cabinet_bottom(55.0, cabinet_zmin),
        ),
        "P2_h85": (
            base["CABINET_XMAX"] + corner_offset,
            base["CABINET_YMIN"] - corner_offset,
            z_from_cabinet_bottom(85.0, cabinet_zmin),
        ),
        "P3_h45": (
            base["CABINET_XMAX"] + det_offset,
            0.0,
            z_from_cabinet_bottom(45.0, cabinet_zmin),
        ),
        "P3_h85": (
            base["CABINET_XMAX"] + det_offset,
            0.0,
            z_from_cabinet_bottom(85.0, cabinet_zmin),
        ),
        "P4_h42": (
            base["CABINET_XMAX"] + corner_offset,
            base["CABINET_YMAX"] + corner_offset,
            z_from_cabinet_bottom(42.0, cabinet_zmin),
        ),
        "P4_h113": (
            base["CABINET_XMAX"] + corner_offset,
            base["CABINET_YMAX"] + corner_offset,
            z_from_cabinet_bottom(113.0, cabinet_zmin),
        ),
        "P5_h45": (
            0.0,
            base["CABINET_YMAX"] + det_offset,
            z_from_cabinet_bottom(45.0, cabinet_zmin),
        ),
        "P5_h85": (
            0.0,
            base["CABINET_YMAX"] + det_offset,
            z_from_cabinet_bottom(85.0, cabinet_zmin),
        ),
    }

    return points


def build_base_neutron_tallies(base, det_size=2.0, energy_bins=None):
    """Build the standard neutron tallies used by the base-model notebook."""

    energy_bins, energy_filter = build_neutron_energy_filter(energy_bins)
    tally_list = []

    # Active source-region tallies.
    source_cell_filter = openmc.CellFilter(base["source_fill_cell"])

    source_flux_spectrum = openmc.Tally(name="source_flux_spectrum")
    source_flux_spectrum.filters = [source_cell_filter, energy_filter]
    source_flux_spectrum.scores = ["flux"]
    tally_list.append(source_flux_spectrum)

    source_flux_total = openmc.Tally(name="source_flux_total")
    source_flux_total.filters = [source_cell_filter]
    source_flux_total.scores = ["flux"]
    tally_list.append(source_flux_total)

    # Region-averaged total flux tallies.
    region_cell_names = [
        ("inner_paraffin_flux_total", "inner_paraffin_cell"),
        ("outer_paraffin_flux_total", "outer_paraffin_cell"),
        ("plastic_flux_total", "plastic_cell"),
        ("cabinet_wall_flux_total", "cabinet_wall_cell"),
        ("cabinet_air_flux_total", "cabinet_air_cell"),
        ("sample1_flux_total", "sample1_cell"),
        ("sample2_flux_total", "sample2_cell"),
    ]

    for tally_name, cell_key in region_cell_names:
        cell = base.get(cell_key)
        if cell is None:
            continue

        cell_filter = openmc.CellFilter(cell)
        tally = openmc.Tally(name=tally_name)
        tally.filters = [cell_filter]
        tally.scores = ["flux"]
        tally_list.append(tally)

    # Local source-position mesh tallies.
    source_position_mesh_size = 0.5
    source_half = source_position_mesh_size / 2.0

    source_position_mesh = openmc.RegularMesh()
    source_position_mesh.dimension = [1, 1, 1]
    source_position_mesh.lower_left = [-source_half, -source_half, -source_half]
    source_position_mesh.upper_right = [+source_half, +source_half, +source_half]

    source_position_mesh_filter = openmc.MeshFilter(source_position_mesh)

    source_position_flux_total = openmc.Tally(name="source_position_flux_total")
    source_position_flux_total.filters = [source_position_mesh_filter]
    source_position_flux_total.scores = ["flux"]
    tally_list.append(source_position_flux_total)

    source_position_flux_spectrum = openmc.Tally(name="source_position_flux_spectrum")
    source_position_flux_spectrum.filters = [source_position_mesh_filter, energy_filter]
    source_position_flux_spectrum.scores = ["flux"]
    tally_list.append(source_position_flux_spectrum)

    # Spatial visualization meshes.
    mesh_xy = openmc.RegularMesh()
    mesh_xy.dimension = [140, 140, 1]
    mesh_xy.lower_left = [base["CABINET_XMIN"] - 5.0, base["CABINET_YMIN"] - 5.0, -1.0]
    mesh_xy.upper_right = [base["CABINET_XMAX"] + 5.0, base["CABINET_YMAX"] + 5.0, 1.0]

    mesh_xy_flux = openmc.Tally(name="spatial_flux_mesh_xy")
    mesh_xy_flux.filters = [openmc.MeshFilter(mesh_xy)]
    mesh_xy_flux.scores = ["flux"]
    tally_list.append(mesh_xy_flux)

    mesh_xz = openmc.RegularMesh()
    mesh_xz.dimension = [140, 1, 160]
    mesh_xz.lower_left = [base["CABINET_XMIN"] - 5.0, -1.0, base["CABINET_ZMIN"] - 5.0]
    mesh_xz.upper_right = [base["CABINET_XMAX"] + 5.0, 1.0, base["CABINET_ZMAX"] + 5.0]

    mesh_xz_flux = openmc.Tally(name="spatial_flux_mesh_xz")
    mesh_xz_flux.filters = [openmc.MeshFilter(mesh_xz)]
    mesh_xz_flux.scores = ["flux"]
    tally_list.append(mesh_xz_flux)

    # Reference P1--P5 point tallies.
    p_tally_points = build_base_tally_points(base)
    point_total_tallies = []
    point_spectrum_tallies = []

    for point_name, center in p_tally_points.items():
        point_total_tallies.append(
            make_mesh_flux_tally(
                name=f"flux_{point_name}",
                center=center,
                det_size=det_size,
                spectral=False,
            )
        )

        point_spectrum_tallies.append(
            make_mesh_flux_tally(
                name=f"flux_spectrum_{point_name}",
                center=center,
                det_size=det_size,
                spectral=True,
                energy_filter=energy_filter,
            )
        )

    tally_list.extend(point_total_tallies)
    tally_list.extend(point_spectrum_tallies)

    tally_data = {
        "tallies": openmc.Tallies(tally_list),
        "energy_bins": energy_bins,
        "energy_filter": energy_filter,
        "P_TALLY_POINTS": p_tally_points,
        "P_SPECTRUM_POINTS": list(p_tally_points.keys()),
        "DET_SIZE": det_size,
        "DET_OFFSET": 1.0,
        "SOURCE_POSITION_MESH_SIZE": source_position_mesh_size,
        "source_position_mesh": source_position_mesh,
        "mesh_xy": mesh_xy,
        "mesh_xz": mesh_xz,
        "point_total_tallies": point_total_tallies,
        "point_spectrum_tallies": point_spectrum_tallies,
    }

    return tally_data


def build_base_neutron_model(
    batches=120,
    particles=50_000,
    inactive=0,
    export_xml=False,
    output_dir=None,
    verbose=False,
    include_cabinet=True,
    include_lead=True,
    source_evaluation_date=None,
):
    """
    Build the complete base neutron fixed-source model with standard tallies.

    This is the compatibility function used by Notebook 1 and the later neutron
    tally notebooks. The OpenMC source is normalized to one source neutron;
    physical scaling is done later using SOURCE_NEUTRON_EMISSION.

    If ``source_evaluation_date`` is supplied, the documented factory emission
    is decay-corrected from 1968-07-05 to that date.  If it is omitted, the
    factory reference-date value is retained and explicitly flagged as not
    decay-corrected.  No source-strength uncertainty is assigned because the
    available factory report does not state one.
    """

    base = build_base_geometry_and_materials(
        verbose=verbose,
        include_cabinet=include_cabinet,
        include_lead=include_lead,
    )

    settings = build_neutron_settings(
        energy_dist=base["energy_dist"],
        batches=batches,
        particles=particles,
        inactive=inactive,
        source_strength=1.0,
    )

    tally_data = build_base_neutron_tallies(base)

    model = openmc.Model(
        geometry=base["geometry"],
        materials=base["materials"],
        settings=settings,
        tallies=tally_data["tallies"],
    )

    model_data = {}
    model_data.update(base)
    model_data.update(tally_data)
    model_data.update(get_source_record(source_evaluation_date))

    model_data["settings"] = settings
    model_data["model"] = model
    model_data["include_cabinet"] = include_cabinet
    model_data["include_lead"] = include_lead

    if verbose:
        print()
        print("Documented physical source:")
        print(f"  Manufacturer:              {model_data['SOURCE_MANUFACTURER']}")
        print(f"  Model:                     {model_data['SOURCE_MODEL']}")
        print(f"  Serial number:             {model_data['SOURCE_SERIAL_NUMBER']}")
        print(
            "  Factory neutron emission:  "
            f"{SOURCE_NEUTRON_EMISSION_REF:.3e} n/s on "
            f"{SOURCE_NEUTRON_EMISSION_REF_DATE}"
        )
        print(
            "  Selected normalization:    "
            f"{model_data['SOURCE_NEUTRON_EMISSION']:.3e} n/s on "
            f"{model_data['SOURCE_EVALUATION_DATE']}"
        )
        print(
            "  Normalization status:      "
            f"{model_data['SOURCE_NORMALIZATION_STATUS']}"
        )
        print(
            "  Source-strength uncertainty: "
            f"{SOURCE_NEUTRON_EMISSION_UNCERTAINTY_STATUS}"
        )
        print(
            "  Historical local thermal flux: "
            f"{HISTORICAL_LOCAL_THERMAL_FLUX:.3e} n/cm2/s"
        )
        print(
            "  Thermal-flux uncertainty:  "
            f"{HISTORICAL_LOCAL_THERMAL_FLUX_UNCERTAINTY_STATUS}"
        )

    if export_xml:
        if output_dir is None:
            model.export_to_xml()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model.export_to_xml(directory=output_dir)

    return model_data


# ============================================================
# Am-Be gamma-line energy distribution
# ============================================================

def build_ambe_gamma_line_distribution(make_plot=False, verbose=False):
    """
    Build an OpenMC photon energy distribution for the prompt Am-Be gamma lines.

    The function uses ALPHANSO when available. The returned distribution is
    normalized to one emitted photon. The absolute photon emission rate must
    be applied separately.
    """

    import numpy as np
    import openmc

    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    try:
        from alphanso.transport import Transport

        config = {
            "name": "AMN 24 Am-Be gamma-spectrum approximation",
            "calc_type": "homogeneous",
            # Legacy composition assumption.  It is not a documented AMN 24
            # active-mixture ratio and must not define the absolute gamma or
            # neutron emission rate.
            "matdef": {
                "Am-241": 0.673,
                "Be-9": 0.327,
            },
            "neutron_energy_bins": [0.0, 12.0, 121],
        }

        results = Transport.calculate(config)

        gamma_yield = float(results["gamma_yield"])
        gamma_lines = results["gamma_lines"]

        gamma_energies_mev = np.array(
            [float(line[0]) for line in gamma_lines],
            dtype=float,
        )

        gamma_line_yields = np.array(
            [float(line[1]) for line in gamma_lines],
            dtype=float,
        )

        source_label = "ALPHANSO Am-Be gamma lines"
        alphanso_results = results

    except Exception as error:
        gamma_energies_mev = np.array([4.4389], dtype=float)
        gamma_line_yields = np.array([1.0], dtype=float)
        gamma_yield = np.nan
        gamma_lines = [[4.4389, 1.0]]
        source_label = f"Fallback Am-Be gamma line model ({error})"
        alphanso_results = None

    gamma_line_yields = np.clip(gamma_line_yields, 0.0, None)

    if gamma_line_yields.sum() <= 0.0:
        raise ValueError("Gamma-line yields contain no positive values.")

    gamma_prob = gamma_line_yields / gamma_line_yields.sum()
    gamma_energies_ev = gamma_energies_mev * 1.0e6

    gamma_energy_dist = openmc.stats.Discrete(
        gamma_energies_ev.tolist(),
        gamma_prob.tolist(),
    )

    gamma_data = {
        "gamma_yield": gamma_yield,
        "gamma_lines": gamma_lines,
        "gamma_energies_mev": gamma_energies_mev,
        "gamma_energies_ev": gamma_energies_ev,
        "gamma_line_yields": gamma_line_yields,
        "gamma_prob": gamma_prob,
        "source_label": source_label,
        "alphanso_results": alphanso_results,
    }

    if verbose:
        print("Am-Be gamma emission lines:")
        print(f"  Total gamma yield: {gamma_yield:.6e}")

        for energy, line_yield, prob in zip(
            gamma_energies_mev,
            gamma_line_yields,
            gamma_prob,
        ):
            print(
                f"  E = {energy:.4f} MeV, "
                f"yield = {line_yield:.6e}, "
                f"probability = {100.0 * prob:.6f} %"
            )

        dominant_index = int(np.argmax(gamma_prob))
        print()
        print("Dominant gamma line:")
        print(f"  E = {gamma_energies_mev[dominant_index]:.4f} MeV")
        print(f"  probability = {100.0 * gamma_prob[dominant_index]:.6f} %")

    if make_plot:
        if plt is None:
            print("Matplotlib is not available, so no gamma-line plot was made.")
        else:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

            ax.bar(
                gamma_energies_mev,
                gamma_prob,
                width=0.08,
                align="center",
                alpha=0.65,
                label=source_label,
            )

            for energy, prob in zip(gamma_energies_mev, gamma_prob):
                ax.text(
                    energy,
                    prob * 1.4,
                    f"{energy:.4f} MeV\n{100.0 * prob:.4f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_yscale("log")
            ax.set_xlabel(r"Gamma energy $E_\gamma$ [MeV]")
            ax.set_ylabel("Relative probability per line")
            ax.set_title("Am-Be gamma emission lines")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            plt.show()

    return gamma_energy_dist, gamma_data