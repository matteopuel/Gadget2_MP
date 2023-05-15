#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>

#include "allvars.h"
#include "proto.h"


/*! \file begrun.c
 *  \brief initial set-up of a simulation run
 *
 *  This file contains various functions to initialize a simulation run. In
 *  particular, the parameterfile is read in and parsed, the initial
 *  conditions or restart files are read, and global variables are
 *  initialized to their proper values.
 */


/*! This function performs the initial set-up of the simulation. First, the
 *  parameterfile is set, then routines for setting units, reading
 *  ICs/restart-files are called, auxialiary memory is allocated, etc.
 */
void begrun(void)
{
  struct global_data_all_processes all;

  if(ThisTask == 0)
    {
      printf("\nThis is Gadget, version `%s'.\n", GADGETVERSION);
      printf("\nRunning on %d processors.\n", NTask);
    }

  read_parameter_file(ParameterFile);	/* ... read in parameters for this run */

  allocate_commbuffers();	/* ... allocate buffer-memory for particle 
				   exchange during force computation */
  set_units();

#if defined(PERIODIC) && (!defined(PMGRID) || defined(FORCETEST))
  ewald_init();
#endif

  open_outputfiles();

  random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);
  gsl_rng_set(random_generator, 42);	/* start-up seed */


// MPUEL: set the random number generator for oscillation, scattering, annihilation
#ifdef ADM_MODEL_ON
  random_generator_ADM = gsl_rng_alloc(gsl_rng_ranlxd1);
  gsl_rng_set(random_generator_ADM, ThisTask+1);  /* start-up seed from MPI rank 
                                                  + 1 in order to avoid collisions with the default settings.
                                                  This means that any processor will have different generator seed */
#endif


#ifdef PMGRID
  long_range_init();
#endif

  All.TimeLastRestartFile = CPUThisRun;

  if(RestartFlag == 0 || RestartFlag == 2)
    {
      set_random_numbers();

      init();			/* ... read in initial model (done only when read IC!) */
    }
  else
    {
      all = All;		/* save global variables. (will be read from restart file) */

      restart(RestartFlag);	/* ... read restart file. Note: This also resets 
				   all variables in the struct `All'. 
				   However, during the run, some variables in the parameter
				   file are allowed to be changed, if desired. These need to 
				   copied in the way below.
				   Note:  All.PartAllocFactor is treated in restart() separately.  
				 */

      All.MinSizeTimestep = all.MinSizeTimestep;
      All.MaxSizeTimestep = all.MaxSizeTimestep;
      All.BufferSize = all.BufferSize;
      All.BunchSizeForce = all.BunchSizeForce;
      All.BunchSizeDensity = all.BunchSizeDensity;
      All.BunchSizeHydro = all.BunchSizeHydro;
      All.BunchSizeDomain = all.BunchSizeDomain;

// MPUEL: add BunchSizeADM
#ifdef ADM_MODEL_ON
#if defined(ANNIHILATION_DM) || defined(SCATTERING_DM)
      All.BunchSizeADM = all.BunchSizeADM;
#endif
#endif

      All.TimeLimitCPU = all.TimeLimitCPU;
      All.ResubmitOn = all.ResubmitOn;
      All.TimeBetSnapshot = all.TimeBetSnapshot;
      All.TimeBetStatistics = all.TimeBetStatistics;
      All.CpuTimeBetRestartFile = all.CpuTimeBetRestartFile;
      All.ErrTolIntAccuracy = all.ErrTolIntAccuracy;
      All.MaxRMSDisplacementFac = all.MaxRMSDisplacementFac;

      All.ErrTolForceAcc = all.ErrTolForceAcc;

      All.TypeOfTimestepCriterion = all.TypeOfTimestepCriterion;
      All.TypeOfOpeningCriterion = all.TypeOfOpeningCriterion;
      All.NumFilesWrittenInParallel = all.NumFilesWrittenInParallel;
      All.TreeDomainUpdateFrequency = all.TreeDomainUpdateFrequency;

      All.SnapFormat = all.SnapFormat;
      All.NumFilesPerSnapshot = all.NumFilesPerSnapshot;
      All.MaxNumNgbDeviation = all.MaxNumNgbDeviation;
      All.ArtBulkViscConst = all.ArtBulkViscConst;


      All.OutputListOn = all.OutputListOn;
      All.CourantFac = all.CourantFac;

      All.OutputListLength = all.OutputListLength;
      memcpy(All.OutputListTimes, all.OutputListTimes, sizeof(double) * All.OutputListLength);


      strcpy(All.ResubmitCommand, all.ResubmitCommand);
      strcpy(All.OutputListFilename, all.OutputListFilename);
      strcpy(All.OutputDir, all.OutputDir);
      strcpy(All.RestartFile, all.RestartFile);
      strcpy(All.EnergyFile, all.EnergyFile);
      strcpy(All.InfoFile, all.InfoFile);
      strcpy(All.CpuFile, all.CpuFile);
      strcpy(All.TimingsFile, all.TimingsFile);
      strcpy(All.SnapshotFileBase, all.SnapshotFileBase);


// MPUEL: copy the information for the velocity and angular tables (they are allowed to be changed, if desired)
#ifdef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION
      strcpy(All.ThetaTableFilename, all.ThetaTableFilename);
      strcpy(All.CrossSectionTableFilename, all.CrossSectionTableFilename);
      All.Nv = all.Nv;
      All.Ntheta = all.Ntheta;
      All.vmin = all.vmin;
      All.vmax = all.vmax;
#endif


      if(All.TimeMax != all.TimeMax)
	readjust_timebase(All.TimeMax, all.TimeMax);
    }

#ifdef PMGRID
  long_range_init_regionsize();
#endif

  if(All.ComovingIntegrationOn)
    init_drift_table(); // here, we build also the CosmicTimeTable[]

  if(RestartFlag == 2)
    All.Ti_nextoutput = find_next_outputtime(All.Ti_Current + 1);
  else
    All.Ti_nextoutput = find_next_outputtime(All.Ti_Current);


  All.TimeLastRestartFile = CPUThisRun;
}




/*! Computes conversion factors between internal code units and the
 *  cgs-system.
 */
void set_units(void)
{
  double meanweight;

  All.UnitTime_in_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
  All.UnitTime_in_Megayears = All.UnitTime_in_s / SEC_PER_MEGAYEAR;

  if(All.GravityConstantInternal == 0)
    All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);
  else
    All.G = All.GravityConstantInternal;

  All.UnitDensity_in_cgs = All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3);
  All.UnitPressure_in_cgs = All.UnitMass_in_g / All.UnitLength_in_cm / pow(All.UnitTime_in_s, 2);
  All.UnitCoolingRate_in_cgs = All.UnitPressure_in_cgs / All.UnitTime_in_s;
  All.UnitEnergy_in_cgs = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);

  /* convert some physical input parameters to internal units */

  All.Hubble = HUBBLE * All.UnitTime_in_s;


// MPUEL: Conversion to code units (note: multiply by the reciprocal units!)
#ifdef ADM_MODEL_ON
  All.c = C / All.UnitVelocity_in_cm_per_s;
#endif

#ifdef OSCILLATION_DM
  //All.delta_m = (All.MajoranaMass * pow(C, 2) * EV) / All.UnitEnergy_in_cgs; // [energy in code units] // wrong!
  All.delta_m = (All.MajoranaMass * EV) * (2.0 * M_PI / PLANCK) * All.UnitTime_in_s; // [1/time in code units]
#endif

#ifdef ANNIHILATION_DM
  All.sigmav_s = All.AnnihilationCrossSectionSwave * All.UnitMass_in_g / pow(All.UnitLength_in_cm, 2) * 1.0e5 / All.UnitVelocity_in_cm_per_s; // 1.0e5 because 1 km = 10^5 cm
#endif

#ifdef SCATTERING_DM
  All.sigma_scatter = All.ScatteringCrossSection * All.UnitMass_in_g / pow(All.UnitLength_in_cm, 2);
#endif


  if(ThisTask == 0)
    {
      printf("\nHubble (internal units) = %g\n", All.Hubble);
      printf("G (internal units) = %g\n", All.G);
      printf("UnitMass_in_g = %g \n", All.UnitMass_in_g);
      printf("UnitTime_in_s = %g \n", All.UnitTime_in_s);
      printf("UnitVelocity_in_cm_per_s = %g \n", All.UnitVelocity_in_cm_per_s);
      printf("UnitDensity_in_cgs = %g \n", All.UnitDensity_in_cgs);
      printf("UnitEnergy_in_cgs = %g \n", All.UnitEnergy_in_cgs);
      printf("\n");


// MPUEL: printf the values of the quantities above
#ifdef ADM_MODEL_ON
      printf("C (internal units) = %g\n", All.c);

#ifdef OSCILLATION_DM
      printf("MajoranaMass (internal units) = %g\n", All.delta_m);
#endif

#ifdef ANNIHILATION_DM
      printf("AnnihilationCrossSectionSwave (internal units) = %g\n", All.sigmav_s);
#endif

#ifdef SCATTERING_DM
      printf("ScatteringCrossSection (internal units) = %g\n", All.sigma_scatter);
#endif

      printf("\n");
#endif // end ADM_MODEL_ON

    }

  meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC);	/* note: we assume neutral gas here */

#ifdef ISOTHERM_EQS
  All.MinEgySpec = 0;
#else
  All.MinEgySpec = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.MinGasTemp;
  All.MinEgySpec *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
#endif

}



/*!  This function opens various log-files that report on the status and
 *   performance of the simulation. On restart from restart-files
 *   (start-option 1), the code will append to these files.
 */
void open_outputfiles(void)
{
  char mode[2], buf[200];

  if(RestartFlag == 0)
    strcpy(mode, "w");
  else
    strcpy(mode, "a");


// MPUEL: write AdmModel txt log-file for the root processor (one by each processor)
#ifdef ADM_MODEL_ON
#if !defined(TEST_HERNQUIST_HALO_SCATT) && !defined(TEST_HERNQUIST_HALO_ANN)
  if(ThisTask == 0) 
  {
#endif
    sprintf(buf, "%sadm_log_%.4d.txt", All.OutputDir, ThisTask); // to write one file for each processor, remove: if(ThisTask == 0){}
    if(!(FdAdmModel = fopen(buf, mode)))
    {
      printf("error in opening file '%s' on core %d\n", buf, ThisTask);
      endrun(1);
    }
#if !defined(TEST_HERNQUIST_HALO_SCATT) && !defined(TEST_HERNQUIST_HALO_ANN)
  }
#endif
#endif // end ADM_MODEL_ON


  if(ThisTask != 0)		/* only the root processor writes to the log files */
    return;

  sprintf(buf, "%s%s", All.OutputDir, All.CpuFile);
  if(!(FdCPU = fopen(buf, mode)))
    {
      printf("error in opening file '%s'\n", buf);
      endrun(1);
    }

  sprintf(buf, "%s%s", All.OutputDir, All.InfoFile);
  if(!(FdInfo = fopen(buf, mode)))
    {
      printf("error in opening file '%s'\n", buf);
      endrun(1);
    }

  sprintf(buf, "%s%s", All.OutputDir, All.EnergyFile);
  if(!(FdEnergy = fopen(buf, mode)))
    {
      printf("error in opening file '%s'\n", buf);
      endrun(1);
    }

  sprintf(buf, "%s%s", All.OutputDir, All.TimingsFile);
  if(!(FdTimings = fopen(buf, mode)))
    {
      printf("error in opening file '%s'\n", buf);
      endrun(1);
    }

#ifdef FORCETEST
  if(RestartFlag == 0)
    {
      sprintf(buf, "%s%s", All.OutputDir, "forcetest.txt");
      if(!(FdForceTest = fopen(buf, "w")))
	{
	  printf("error in opening file '%s'\n", buf);
	  endrun(1);
	}
      fclose(FdForceTest);
    }
#endif
}


/*!  This function closes the global log-files.
 */
void close_outputfiles(void)
{

// MPUEL: close FdAdmModel file by every processor
#ifdef ADM_MODEL_ON
  fclose(FdAdmModel);
#endif 

  if(ThisTask != 0)		/* only the root processor writes to the log files */
    return;

  fclose(FdCPU);
  fclose(FdInfo);
  fclose(FdEnergy);
  fclose(FdTimings);
#ifdef FORCETEST
  fclose(FdForceTest);
#endif
}




/*! This function parses the parameterfile in a simple way.  Each paramater
 *  is defined by a keyword (`tag'), and can be either of type double, int,
 *  or character string.  The routine makes sure that each parameter
 *  appears exactly once in the parameterfile, otherwise error messages are
 *  produced that complain about the missing parameters.
 */
void read_parameter_file(char *fname)
{
#define DOUBLE 1
#define STRING 2
#define INT 3
#define MAXTAGS 300

  FILE *fd, *fdout;
  char buf[200], buf1[200], buf2[200], buf3[400];
  int i, j, nt;
  int id[MAXTAGS];
  void *addr[MAXTAGS];
  char tag[MAXTAGS][50];
  int  errorFlag = 0;


  if(sizeof(long long) != 8)
    {
      if(ThisTask == 0)
	printf("\nType `long long' is not 64 bit on this platform. Stopping.\n\n");
      endrun(0);
    }

  if(sizeof(int) != 4)
    {
      if(ThisTask == 0)
	printf("\nType `int' is not 32 bit on this platform. Stopping.\n\n");
      endrun(0);
    }

  if(sizeof(float) != 4)
    {
      if(ThisTask == 0)
	printf("\nType `float' is not 32 bit on this platform. Stopping.\n\n");
      endrun(0);
    }

  if(sizeof(double) != 8)
    {
      if(ThisTask == 0)
	printf("\nType `double' is not 64 bit on this platform. Stopping.\n\n");
      endrun(0);
    }


// MPUEL: add compilation errors if ADM_MODEL_ON is defined
#ifdef ADM_MODEL_ON

#if defined(VECTOR_MEDIATOR) && defined(SCALAR_MEDIATOR)
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with VECTOR_MEDIATOR and with SCALAR_MEDIATOR.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif

#if defined(TRANSFER_CROSS_SECTION) && !defined(SCATTERING_DM)
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with TRANSFER_CROSS_SECTION, but not with SCATTERING_DM.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif

#if defined(MODIFIED_TRANSFER_CROSS_SECTION) && !defined(SCATTERING_DM)
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with MODIFIED_TRANSFER_CROSS_SECTION, but not with SCATTERING_DM.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif

#if defined(VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION) && !defined(SCATTERING_DM)
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION, but not with SCATTERING_DM.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif

#if defined(OUTPUTDT_SCATTER) && !defined(SCATTERING_DM)
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with OUTPUTDT_SCATTER, but not with SCATTERING_DM.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif

#else // if ADM_MODEL_ON is not defined

#ifdef VECTOR_MEDIATOR
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with VECTOR_MEDIATOR, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end VECTOR_MEDIATOR

#ifdef SCALAR_MEDIATOR
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with SCALAR_MEDIATOR, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end SCALAR_MEDIATOR

#ifdef OSCILLATION_DM
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with OSCILLATION_DM, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end OSCILLATION_DM

#ifdef ANNIHILATION_DM
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with ANNIHILATION_DM, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end ANNIHILATION_DM

#ifdef SCATTERING_DM
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with SCATTERING_DM, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end SCATTERING_DM

#ifdef TRANSFER_CROSS_SECTION
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with TRANSFER_CROSS_SECTION, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end TRANSFER_CROSS_SECTION

#ifdef MODIFIED_TRANSFER_CROSS_SECTION
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with MODIFIED_TRANSFER_CROSS_SECTION, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end MODIFIED_TRANSFER_CROSS_SECTION

#ifdef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION

#ifdef OUTPUTDT_SCATTER
  if(ThisTask == 0)
    {
      fprintf(stdout, "Code was compiled with OUTPUTDT_SCATTER, but not with ADM_MODEL_ON.\n");
      fprintf(stdout, "This is not allowed.\n");
    }
  endrun(0);
#endif // end OUTPUTDT_SCATTER

#endif // end ADM_MODEL_ON


  if(ThisTask == 0)		/* read parameter file on process 0 */
    {
      nt = 0;

      strcpy(tag[nt], "InitCondFile");
      addr[nt] = All.InitCondFile;
      id[nt++] = STRING;

      strcpy(tag[nt], "OutputDir");
      addr[nt] = All.OutputDir;
      id[nt++] = STRING;

      strcpy(tag[nt], "SnapshotFileBase");
      addr[nt] = All.SnapshotFileBase;
      id[nt++] = STRING;

      strcpy(tag[nt], "EnergyFile");
      addr[nt] = All.EnergyFile;
      id[nt++] = STRING;

      strcpy(tag[nt], "CpuFile");
      addr[nt] = All.CpuFile;
      id[nt++] = STRING;

      strcpy(tag[nt], "InfoFile");
      addr[nt] = All.InfoFile;
      id[nt++] = STRING;

      strcpy(tag[nt], "TimingsFile");
      addr[nt] = All.TimingsFile;
      id[nt++] = STRING;

      strcpy(tag[nt], "RestartFile");
      addr[nt] = All.RestartFile;
      id[nt++] = STRING;

      strcpy(tag[nt], "ResubmitCommand");
      addr[nt] = All.ResubmitCommand;
      id[nt++] = STRING;

      strcpy(tag[nt], "OutputListFilename");
      addr[nt] = All.OutputListFilename;
      id[nt++] = STRING;

      strcpy(tag[nt], "OutputListOn");
      addr[nt] = &All.OutputListOn;
      id[nt++] = INT;

      strcpy(tag[nt], "Omega0");
      addr[nt] = &All.Omega0;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "OmegaBaryon");
      addr[nt] = &All.OmegaBaryon;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "OmegaLambda");
      addr[nt] = &All.OmegaLambda;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "HubbleParam");
      addr[nt] = &All.HubbleParam;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "BoxSize");
      addr[nt] = &All.BoxSize;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "PeriodicBoundariesOn");
      addr[nt] = &All.PeriodicBoundariesOn;
      id[nt++] = INT;

      strcpy(tag[nt], "TimeOfFirstSnapshot");
      addr[nt] = &All.TimeOfFirstSnapshot;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "CpuTimeBetRestartFile");
      addr[nt] = &All.CpuTimeBetRestartFile;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "TimeBetStatistics");
      addr[nt] = &All.TimeBetStatistics;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "TimeBegin");
      addr[nt] = &All.TimeBegin;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "TimeMax");
      addr[nt] = &All.TimeMax;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "TimeBetSnapshot");
      addr[nt] = &All.TimeBetSnapshot;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "UnitVelocity_in_cm_per_s");
      addr[nt] = &All.UnitVelocity_in_cm_per_s;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "UnitLength_in_cm");
      addr[nt] = &All.UnitLength_in_cm;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "UnitMass_in_g");
      addr[nt] = &All.UnitMass_in_g;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "TreeDomainUpdateFrequency");
      addr[nt] = &All.TreeDomainUpdateFrequency;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "ErrTolIntAccuracy");
      addr[nt] = &All.ErrTolIntAccuracy;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "ErrTolTheta");
      addr[nt] = &All.ErrTolTheta;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "ErrTolForceAcc");
      addr[nt] = &All.ErrTolForceAcc;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "MinGasHsmlFractional");
      addr[nt] = &All.MinGasHsmlFractional;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "MaxSizeTimestep");
      addr[nt] = &All.MaxSizeTimestep;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "MinSizeTimestep");
      addr[nt] = &All.MinSizeTimestep;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "MaxRMSDisplacementFac");
      addr[nt] = &All.MaxRMSDisplacementFac;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "ArtBulkViscConst");
      addr[nt] = &All.ArtBulkViscConst;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "CourantFac");
      addr[nt] = &All.CourantFac;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "DesNumNgb");
      addr[nt] = &All.DesNumNgb;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "MaxNumNgbDeviation");
      addr[nt] = &All.MaxNumNgbDeviation;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "ComovingIntegrationOn");
      addr[nt] = &All.ComovingIntegrationOn;
      id[nt++] = INT;

      strcpy(tag[nt], "ICFormat");
      addr[nt] = &All.ICFormat;
      id[nt++] = INT;

      strcpy(tag[nt], "SnapFormat");
      addr[nt] = &All.SnapFormat;
      id[nt++] = INT;

      strcpy(tag[nt], "NumFilesPerSnapshot");
      addr[nt] = &All.NumFilesPerSnapshot;
      id[nt++] = INT;

      strcpy(tag[nt], "NumFilesWrittenInParallel");
      addr[nt] = &All.NumFilesWrittenInParallel;
      id[nt++] = INT;

      strcpy(tag[nt], "ResubmitOn");
      addr[nt] = &All.ResubmitOn;
      id[nt++] = INT;

      strcpy(tag[nt], "TypeOfTimestepCriterion");
      addr[nt] = &All.TypeOfTimestepCriterion;
      id[nt++] = INT;

      strcpy(tag[nt], "TypeOfOpeningCriterion");
      addr[nt] = &All.TypeOfOpeningCriterion;
      id[nt++] = INT;

      strcpy(tag[nt], "TimeLimitCPU");
      addr[nt] = &All.TimeLimitCPU;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningHalo");
      addr[nt] = &All.SofteningHalo;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningDisk");
      addr[nt] = &All.SofteningDisk;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningBulge");
      addr[nt] = &All.SofteningBulge;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningGas");
      addr[nt] = &All.SofteningGas;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningStars");
      addr[nt] = &All.SofteningStars;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningBndry");
      addr[nt] = &All.SofteningBndry;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningHaloMaxPhys");
      addr[nt] = &All.SofteningHaloMaxPhys;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningDiskMaxPhys");
      addr[nt] = &All.SofteningDiskMaxPhys;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningBulgeMaxPhys");
      addr[nt] = &All.SofteningBulgeMaxPhys;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningGasMaxPhys");
      addr[nt] = &All.SofteningGasMaxPhys;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningStarsMaxPhys");
      addr[nt] = &All.SofteningStarsMaxPhys;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "SofteningBndryMaxPhys");
      addr[nt] = &All.SofteningBndryMaxPhys;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "BufferSize");
      addr[nt] = &All.BufferSize;
      id[nt++] = INT;

      strcpy(tag[nt], "PartAllocFactor");
      addr[nt] = &All.PartAllocFactor;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "TreeAllocFactor");
      addr[nt] = &All.TreeAllocFactor;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "GravityConstantInternal");
      addr[nt] = &All.GravityConstantInternal;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "InitGasTemp");
      addr[nt] = &All.InitGasTemp;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "MinGasTemp");
      addr[nt] = &All.MinGasTemp;
      id[nt++] = DOUBLE;


// MPUEL: model parameters
#if defined(VECTOR_MEDIATOR) || defined(SCALAR_MEDIATOR)
      strcpy(tag[nt], "MediatorDMmassratio");
      addr[nt] = &All.MediatorDMmassratio;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "VectorOrScalarCoupling");
      addr[nt] = &All.VectorOrScalarCoupling;
      id[nt++] = DOUBLE;
#endif

#ifdef OSCILLATION_DM
      strcpy(tag[nt], "MajoranaMass");
      addr[nt] = &All.MajoranaMass;
      id[nt++] = DOUBLE;
#endif

#ifdef ANNIHILATION_DM
      strcpy(tag[nt], "AnnihilationCrossSectionSwave");
      addr[nt] = &All.AnnihilationCrossSectionSwave;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "AnnihilateSearchRadius");
      addr[nt] = &All.AnnihilateSearchRadius;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "AnnihilateSearchRadiusMaxPhys");
      addr[nt] = &All.AnnihilateSearchRadiusMaxPhys;
      id[nt++] = DOUBLE;
#endif

#ifdef SCATTERING_DM
      strcpy(tag[nt], "ScatteringCrossSection");
      addr[nt] = &All.ScatteringCrossSection;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "ScatterSearchRadius");
      addr[nt] = &All.ScatterSearchRadius;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "ScatterSearchRadiusMaxPhys");
      addr[nt] = &All.ScatterSearchRadiusMaxPhys;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "ProbabilityTol");
      addr[nt] = &All.ProbabilityTol;
      id[nt++] = DOUBLE;
#endif

#ifdef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION
      strcpy(tag[nt], "ThetaTableFilename");
      addr[nt] = All.ThetaTableFilename;
      id[nt++] = STRING;

      strcpy(tag[nt], "CrossSectionTableFilename");
      addr[nt] = All.CrossSectionTableFilename;
      id[nt++] = STRING;

      strcpy(tag[nt], "Nv");
      addr[nt] = &All.Nv;
      id[nt++] = INT;

      strcpy(tag[nt], "Ntheta");
      addr[nt] = &All.Ntheta;
      id[nt++] = INT;

      strcpy(tag[nt], "vmin");
      addr[nt] = &All.vmin;
      id[nt++] = DOUBLE;

      strcpy(tag[nt], "vmax");
      addr[nt] = &All.vmax;
      id[nt++] = DOUBLE;
#endif


      if((fd = fopen(fname, "r")))
	{
	  sprintf(buf, "%s%s", fname, "-usedvalues");
	  if(!(fdout = fopen(buf, "w")))
	    {
	      printf("error opening file '%s' \n", buf);
	      errorFlag = 1;
	    }
	  else
	    {
	      while(!feof(fd))
		{
		  *buf = 0;
		  fgets(buf, 200, fd);
		  if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
		    continue;

		  if(buf1[0] == '%')
		    continue;

		  for(i = 0, j = -1; i < nt; i++)
		    if(strcmp(buf1, tag[i]) == 0)
		      {
			j = i;
			tag[i][0] = 0;
			break;
		      }

		  if(j >= 0)
		    {
		      switch (id[j])
			{
			case DOUBLE:
			  *((double *) addr[j]) = atof(buf2);
			  fprintf(fdout, "%-35s%g\n", buf1, *((double *) addr[j]));
			  break;
			case STRING:
			  strcpy(addr[j], buf2);
			  fprintf(fdout, "%-35s%s\n", buf1, buf2);
			  break;
			case INT:
			  *((int *) addr[j]) = atoi(buf2);
			  fprintf(fdout, "%-35s%d\n", buf1, *((int *) addr[j]));
			  break;
			}
		    }
		  else
		    {
		      fprintf(stdout, "Error in file %s:   Tag '%s' not allowed or multiple defined.\n",
			      fname, buf1);
		      errorFlag = 1;
		    }
		}
	      fclose(fd);
	      fclose(fdout);

	      i = strlen(All.OutputDir);
	      if(i > 0)
		if(All.OutputDir[i - 1] != '/')
		  strcat(All.OutputDir, "/");

	      sprintf(buf1, "%s%s", fname, "-usedvalues");
	      sprintf(buf2, "%s%s", All.OutputDir, "parameters-usedvalues");
	      sprintf(buf3, "cp %s %s", buf1, buf2);
	      system(buf3);
	    }
	}
      else
	{
	  printf("\nParameter file %s not found.\n\n", fname);
	  errorFlag = 2;
	}

      if(errorFlag != 2)
	for(i = 0; i < nt; i++)
	  {
	    if(*tag[i])
	      {
		printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], fname);
		errorFlag = 1;
	      }
	  }

      if(All.OutputListOn && errorFlag == 0)
	errorFlag += read_outputlist(All.OutputListFilename);
      else
	All.OutputListLength = 0;


// MPUEL: read scattering angle and cross section table (add defined(ADM_MODEL_ON) because we define the functions in adm_model.c)
#if defined(ADM_MODEL_ON) && defined(VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
  errorFlag += read_thetatable(All.ThetaTableFilename);
  errorFlag += read_crosssectiontable(All.CrossSectionTableFilename);
#endif


    }

  MPI_Bcast(&errorFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(errorFlag)
    {
      MPI_Finalize();
      exit(0);
    }

  /* now communicate the relevant parameters to the other processes */
  MPI_Bcast(&All, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MPI_COMM_WORLD);


  if(All.NumFilesWrittenInParallel < 1)
    {
      if(ThisTask == 0)
	printf("NumFilesWrittenInParallel MUST be at least 1\n");
      endrun(0);
    }

  if(All.NumFilesWrittenInParallel > NTask)
    {
      if(ThisTask == 0)
	printf("NumFilesWrittenInParallel MUST be smaller than number of processors\n");
      endrun(0);
    }

#ifdef PERIODIC
  if(All.PeriodicBoundariesOn == 0)
    {
      if(ThisTask == 0)
	{
	  printf("Code was compiled with periodic boundary conditions switched on.\n");
	  printf("You must set `PeriodicBoundariesOn=1', or recompile the code.\n");
	}
      endrun(0);
    }
#else
  if(All.PeriodicBoundariesOn == 1)
    {
      if(ThisTask == 0)
	{
	  printf("Code was compiled with periodic boundary conditions switched off.\n");
	  printf("You must set `PeriodicBoundariesOn=0', or recompile the code.\n");
	}
      endrun(0);
    }
#endif


  if(All.TypeOfTimestepCriterion >= 1)
    {
      if(ThisTask == 0)
	{
	  printf("The specified timestep criterion\n");
	  printf("is not valid\n");
	}
      endrun(0);
    }

#if defined(LONG_X) ||  defined(LONG_Y) || defined(LONG_Z)
#ifndef NOGRAVITY
  if(ThisTask == 0)
    {
      printf("Code was compiled with LONG_X/Y/Z, but not with NOGRAVITY.\n");
      printf("Stretched periodic boxes are not implemented for gravity yet.\n");
    }
  endrun(0);
#endif
#endif

#undef DOUBLE
#undef STRING
#undef INT
#undef MAXTAGS
}


/*! this function reads a table with a list of desired output times. The
 *  table does not have to be ordered in any way, but may not contain more
 *  than MAXLEN_OUTPUTLIST entries.
 */
int read_outputlist(char *fname)
{
  FILE *fd;

  if(!(fd = fopen(fname, "r")))
    {
      printf("can't read output list in file '%s'\n", fname);
      return 1;
    }

  All.OutputListLength = 0;
  do
    {
      if(fscanf(fd, " %lg ", &All.OutputListTimes[All.OutputListLength]) == 1)
	All.OutputListLength++;
      else
	break;
    }
  while(All.OutputListLength < MAXLEN_OUTPUTLIST);

  fclose(fd);

  printf("\nfound %d times in output-list.\n", All.OutputListLength);

  return 0;
}


/*! If a restart from restart-files is carried out where the TimeMax
 *  variable is increased, then the integer timeline needs to be
 *  adjusted. The approach taken here is to reduce the resolution of the
 *  integer timeline by factors of 2 until the new final time can be
 *  reached within TIMEBASE.
 */
void readjust_timebase(double TimeMax_old, double TimeMax_new)
{
  int i;
  long long ti_end;

  if(ThisTask == 0)
    {
      printf("\nAll.TimeMax has been changed in the parameterfile\n");
      printf("Need to adjust integer timeline\n\n\n");
    }

  if(TimeMax_new < TimeMax_old)
    {
      if(ThisTask == 0)
	printf("\nIt is not allowed to reduce All.TimeMax\n\n");
      endrun(556);
    }

  if(All.ComovingIntegrationOn)
    ti_end = log(TimeMax_new / All.TimeBegin) / All.Timebase_interval;
  else
    ti_end = (TimeMax_new - All.TimeBegin) / All.Timebase_interval;

  while(ti_end > TIMEBASE)
    {
      All.Timebase_interval *= 2.0;

      ti_end /= 2;
      All.Ti_Current /= 2;

#ifdef PMGRID
      All.PM_Ti_begstep /= 2;
      All.PM_Ti_endstep /= 2;
#endif

      for(i = 0; i < NumPart; i++)
	{
	  P[i].Ti_begstep /= 2;
	  P[i].Ti_endstep /= 2;
	}
    }

  All.TimeMax = TimeMax_new;
}

