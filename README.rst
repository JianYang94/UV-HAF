UV HAF analysis
===============

This repository contains the analysis and figure-generation workflow for the
assessment of erythemally weighted ultraviolet radiation in a composite
terrestrial hazard framework.

Repository contents
-------------------

``code`` contains the climate and ocean emulator, ultraviolet reconstruction,
spectral-emulator training programs, composite score calculations, sensitivity
analyses, threshold-crossing diagnostics and figure-generation programs.
``data/derived`` contains the serialized spectral emulator, its metadata, the
biological action spectrum, final CRITIC weights and source data for the figures.
The released Water Hazard Index and land mask are stored beside the code that
loads them. ``figures`` contains the final main and supplementary figures.
``environment`` contains the software requirements.

External input data
-------------------

Monthly UKESM1-0-LL output for ensemble member r1i1p1f2 is required for
SSP1-2.6, SSP2-4.5 and SSP5-8.5 over 2015 to 2100. The input directory is set
with the ``UV_HAF_UKESM_ROOT`` environment variable. Each scenario directory
contains subdirectories for the monthly variables listed in the supplementary
input table. The code reads the original NetCDF files and writes annual UV
caches to ``outputs``.

The climate and ocean emulator downloads and caches HadCRUT5 global mean
surface temperature, NOAA/NCEI 0 to 2000 m ocean heat content and IPCC AR6
effective radiative forcing for 1750 to 2019. Historical forcing is extended
after 2019 by interpolation between the total-forcing anchors for the selected
SSP. Embedded CO2 concentration anchors provide the ocean-chemistry pathway.
The processed Water Hazard Index and Natural Earth land mask are distributed
with the workflow. The Water Hazard Index supplies a fixed spatial baseline and
does not enter the CRITIC indicator weighting.

Ultraviolet reconstruction inputs
---------------------------------

The UV calculation reads the monthly UKESM1 variables ``rsds``, ``rsdscs``,
``rsuscs``, ``huss``, ``ps``, ``od550aer`` and three-dimensional ``o3``. The
ozone files also supply hybrid-height bounds and surface orography. The released
spectral emulator is the 19-component targeted version trained on 2,816
libRadtran cases and assessed against 768 independent radiative-transfer cases.
Its SHA-256 digest is recorded in the data manifest. Erythemal weighting uses
the released 1 nm CIE action-spectrum table. Daily exposure was calculated at
24 hourly local-solar-time midpoints for one representative day per month and
combined using calendar-month lengths.

Analysis sequence
-----------------

1. Install the packages listed in ``environment/requirements.txt``.
2. Set ``UV_HAF_UKESM_ROOT`` to the local UKESM1-0-LL input directory.
3. Reconstruct annual all-sky erythemal UV for the three pathways.
4. Estimate the independent CHS6 and CHS7 weights from the smoothed SSP2-4.5
   training series and calculate the fixed reference thresholds.
5. Calculate CHS, HAF, the direct UV comparison, sensitivity tests and first
   persistent threshold crossing.
6. Generate the five main figures and four supplementary figures.

Methodological order
--------------------

The workflow follows the order used in the Methods: input preparation,
standardisation of the six climate and ocean indicators, reconstruction and
standardisation of erythemal UV, independent weighting of CHS6 and CHS7,
calculation of HAF and persistent threshold crossing, sensitivity analyses and
single-factor radiative diagnostics. Annual display curves use centred 11 year
smoothing with edge padding. Period maps use unsmoothed annual fields averaged
over 2041 to 2060 or 2081 to 2100. Antarctica is displayed for geographic
continuity and excluded from thresholds, land-area-weighted means and HAF.

Data products
-------------

The source data supplied for each figure reproduce the displayed values without
requiring redistribution of the original CMIP6 files. Figure S4 source fields
include the six counterfactual maps and nine-point trajectories used in the
display. The annual UV cache is an intermediate product and can be regenerated
from the public UKESM1-0-LL input. Released figure-source data, weights and model
metadata are read from ``data/derived`` when regenerated intermediate outputs
are not present.

