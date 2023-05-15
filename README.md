### GADGET-2 code with Oscillating ADM model (for ligth scalar and vector mediators) 

Code used in:

Cline, J. M., Gambini, G., McDermott S. D., Puel, M., *Late-Time Dark Matter Oscillations and the Core-Cusp Problem*, J. High Energ. Phys. 2021, 223 (2021), https://doi.org/10.1007/JHEP04(2021)223

and based on GADGET-2, written by Volker Springel (https://wwwmpa.mpa-garching.mpg.de/gadget/). Please cite the initial paper, bseides the reference above, if using this code.

Author of modifications: Matteo Puel
Year: 2020


## Compilation 
The compilation file is **Makefile**, which is the same as in GADGET-2 (see corresponding manual). New parameters have been added and here below we provide a description.

ADM_MODEL_ON : Master flag to allow to turn on our model properties.
- VECTOR_MEDIATOR : Massive light vector mediator interaction. It includes the Sommerfeld enhancement factor for s-wave annihilation
- SCALAR_MEDIATOR : Massive light scalar mediator interaction. It includes the Sommerfeld enhancement factor for s-wave annihilation
- OSCILLATION_DM : Activate oscillations
- ANNIHILATION_DM : Activate annihilations (just s-wave cross section). It requires oscillation probability for Dirac DM. 
- SCATTERING_DM : Activate scatterings. Standard is the viscosity cross section with isotropic scattering
-TRANSFER_CROSS_SECTION : Requires SCATTERING_DM to be active! Replace the viscosity cross section with the original transfer cross section. Still isotropic scattering
- MODIFIED_TRANSFER_CROSS_SECTION : Requires SCATTERING_DM to be active! Replace the viscosity cross section with the modified transfer cross section. Still isotropic scattering
- VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION : Requires SCATTERING_DM to be active! Polar angle and cross section values stored in tables given in input. Anisotropic scattering
- LATE_TIME_PHASE_CHANGE : Requires SCATTERING_DM to be active! Add random phase of scattered particles if cosmic time > 1/MajoranaMass
- OUTPUTDT_SCATTER : Requires SCATTERING_DM to be active! Include the current dt_scatter variable of all particles in the snapshot files (similar to OUTPUTTIMESTEP). dt_scatter is also read from the snapshot if simulation has RestartFlag == 2
- TEST_UNIFORM_BKG : Test 1: scattering of cube to uniform background (no gravity, just one scattering per each particle)
- TEST_HERNQUIST_HALO_SCATT : Test 2: scattering in an isolated Hernquist halo (no change in the particle velocity)
- TEST_HERNQUIST_HALO_ANN : Test 2: scattering in an isolated Hernquist halo (no remove annihilated particles)


Additional paramterfile options:
- MediatorDMmassratio : Mass ratio between mediator and DM, equivalent to dimensionless w (only for VECTOR_MEDIATOR)
- VectorOrScalarCoupling : Dimensionless coupling between vector or scalar mediator and DM, equivalent to \alpha_D
- MajoranaMass : Majorana mass delta_m in eV causing DM oscillations (only for OSCILLATION_DM)
- AnnihilationCrossSectionSwave : The DM velocity-averaged annihilation cross section (s-wave, i.e constant, velocity-independent) per unit DM mass in cm^3/g/sec
- AnnihilateSearchRadius : The neighbour search radius for DM annihilation (h_A) in comoving code length units
- AnnihilateSearchRadiusMaxPhys : The maximum physical search radius (a \times h_A) for DM annihilation in code length units
- ScatteringCrossSection : The DM scattering cross section per unit mass in cm^2/g
- ScatterSearchRadius : The neighbour search radius for DM scattering (h_S) in comoving code length units
- ScatterSearchRadiusMaxPhys : The maximum physical search radius (a \times h_S) for DM scattering in code length units
- ThetaTableFilename : Name of the file containing polar angles drawn from an anisotropic velocity-dependent differential cross section (only for VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
- CrossSectionTableFilename : Name of the file containing total cross sections drawn from an anisotropic velocity-dependent differential cross section as a function of velocity (only for VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
- Nv : The number of velocities at which the cross-section and a sample of thetas are defined (only for VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
- Ntheta : The number of samples drawn from the differential cross-section at each velocity (only for VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
- vmin : The minumum velocity (in km/s) at which the cross-section is defined. For velocities v<vmin, the cross-section is set equal to its value at vmin (only for VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
- vmax : The maximum velocity (in km/s) at which the cross-section is defined. For velocities v>vmax, the cross-section is set to 0 (only for VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
- ProbabilityTol : Maximum probability of scattering, used to introduce a DM scattering limiter (not for annihilation because the two annihilating particles disappear)

## Bibtex
@article{Cline:2020gon,
    author = "Cline, James M. and Gambini, Guillermo and Mcdermott, Samuel D. and Puel, Matteo",
    title = "{Late-Time Dark Matter Oscillations and the Core-Cusp Problem}",
    eprint = "2010.12583",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "FNAL-PUB-20-556-T, FERMILAB-PUB-20-556-T",
    doi = "10.1007/JHEP04(2021)223",
    journal = "JHEP",
    volume = "04",
    pages = "223",
    year = "2021"
}

