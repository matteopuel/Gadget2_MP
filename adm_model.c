// Author: MPUEL
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "allvars.h"
#include "proto.h"


/*! \file adm_model.c
 *  \brief Find neighbours, and perform Monte-Carlo Scattering and Annihilation.
 *  Perform oscillation as well.
 */

#ifdef ADM_MODEL_ON 


// MPUEL: add integrated function and get_cosmic_time()
double get_cosmic_time(double a0) // a0 will be All.Time in the code
{
  double a1p, a1m, t1, u1;
  int i1;

  /* note: will only be called for cosmological integration */
  /* MPUEL: linear interpolation is a very good approximation looking 
   * at the plot t(a) vs. a when we integrate cosmictime_integ() numerically*/

  u1 = (a0 - All.TimeBegin) / (All.TimeMax - All.TimeBegin) * COSMIC_TIME_TABLE_LENGTH;
  i1 = (int) u1; // = i + 1
  if(i1 >= COSMIC_TIME_TABLE_LENGTH)
    i1 = COSMIC_TIME_TABLE_LENGTH - 1; // = i

  if(i1 <= 1)
    t1 = CosmicTimeTable[0]; 
  else
  {
    a1p = All.TimeBegin + (All.TimeMax - All.TimeBegin) / COSMIC_TIME_TABLE_LENGTH * i1; // a(i1)
    a1m = All.TimeBegin + (All.TimeMax - All.TimeBegin) / COSMIC_TIME_TABLE_LENGTH * (i1 - 1); // a(i1 - 1)
    t1 = CosmicTimeTable[i1 - 1] + (CosmicTimeTable[i1] - CosmicTimeTable[i1 - 1]) / (a1p - a1m) * (a0 - a1m); // linear interpolation
  }

  return t1;
}


double cosmictime_integ(double a, void *param) // used in driftfac.c
{
  double h;

  h = All.Omega0 / (a * a * a) + (1 - All.Omega0 - All.OmegaLambda) / (a * a) + All.OmegaLambda;
  h = All.Hubble * All.HubbleParam * sqrt(h);

  return 1 / (h * a); // equivalent to da/(a H)
}



#if defined(ANNIHILATION_DM) || defined(SCATTERING_DM)



#ifdef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION

/*! this function reads a table with a list of scattering angles, theta,
 *  spaced according to the angularly-dependent cross-section
 *  being simulated.
 */
int read_thetatable(char *fname) {
  FILE *fd;
  int count, flag;
  char buf[512];

  if (!(fd = fopen(fname, "r"))) 
  {
    printf("can't read theta table in file '%s'\n", fname);
    return 1;
  }

  All.ThetaTableLength = 0;

  while (1) 
  {
    if (fgets(buf, 500, fd) != buf) break;

    count = sscanf(buf, " %lg %d ", &All.ThetaTable[All.ThetaTableLength], &flag);

    if (count == 1) flag = 1;

    if (count == 1 || count == 2) 
    {
      if (All.ThetaTableLength >= MAXLEN_THETA_TABLE) 
      {
        if (ThisTask == 0)
          printf("\ntoo many entries in theta table. You should increase "
              "MAXLEN_THETA_TABLE=%d.\n", (int)MAXLEN_THETA_TABLE);
        
        endrun(13);
      }

      All.ThetaTableFlag[All.ThetaTableLength] = flag;
      All.ThetaTableLength++;
    }
  }

  fclose(fd);

  printf("\nfound %d angles in theta table.\n", All.ThetaTableLength);

  if (All.ThetaTableLength != All.Nv * All.Ntheta) 
  {  
    if (ThisTask == 0)
      printf("\nSize of thetable is not equal to Nv*Ntheta");
      
    return 1;
  }

  return 0;
}


/*! this function reads a table with a list of cross-sections
 *  as a function of velocity. These are for Nv different velocities
 *  logarithmically spaced from vmin to vmax
 */
int read_crosssectiontable(char *fname) {
  FILE *fd;
  int count, flag;
  char buf[512];

  if (!(fd = fopen(fname, "r"))) 
  {
    printf("can't read cross-section table in file '%s'\n", fname);
    return 1;
  }

  All.CrossSectionTableLength = 0;

  while (1) 
  {
    if (fgets(buf, 500, fd) != buf) break;

    count = sscanf(buf, " %lg %d ", &All.CrossSectionTable[All.CrossSectionTableLength], &flag);

    if (count == 1) flag = 1;

    if (count == 1 || count == 2) 
    {
      if (All.CrossSectionTableLength >= MAXLEN_CROSSSECTION_TABLE) 
      {
        if (ThisTask == 0)
          printf("\ntoo many entries in cross-section table. You should increase "
              "MAXLEN_CROSSSECTION_TABLE=%d.\n", (int)MAXLEN_CROSSSECTION_TABLE);

        endrun(13);
      }

      All.CrossSectionTableFlag[All.CrossSectionTableLength] = flag;
      All.CrossSectionTableLength++;
    }
  }

  fclose(fd);

  printf("\nfound %d angles in cross-section table.\n", All.CrossSectionTableLength);

  if (All.CrossSectionTableLength != All.Nv) 
  {  
    if (ThisTask == 0)
      printf("\nSize of cross-section table is not equal to Nv");
       
    return 1;
  }

  return 0;
}

#endif // end VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION



#ifdef PERIODIC
static double boxSize, boxHalf;

#ifdef LONG_X
static double boxSize_X, boxHalf_X;
#else
#define boxSize_X boxSize
#define boxHalf_X boxHalf
#endif

#ifdef LONG_Y
static double boxSize_Y, boxHalf_Y;
#else
#define boxSize_Y boxSize
#define boxHalf_Y boxHalf
#endif

#ifdef LONG_Z
static double boxSize_Z, boxHalf_Z;
#else
#define boxSize_Z boxSize
#define boxHalf_Z boxHalf
#endif

#endif // end PERIODIC


/*! these macros maps a coordinate difference to the nearest periodic
 * image
 */

#define NGB_PERIODIC_X(x) (xtmp=(x),(xtmp>boxHalf_X)?(xtmp-boxSize_X):((xtmp<-boxHalf_X)?(xtmp+boxSize_X):xtmp))
#define NGB_PERIODIC_Y(x) (xtmp=(x),(xtmp>boxHalf_Y)?(xtmp-boxSize_Y):((xtmp<-boxHalf_Y)?(xtmp+boxSize_Y):xtmp))
#define NGB_PERIODIC_Z(x) (xtmp=(x),(xtmp>boxHalf_Z)?(xtmp-boxSize_Z):((xtmp<-boxHalf_Z)?(xtmp+boxSize_Z):xtmp))



/*! Inspired by density() in density.c
*/
void annihilate_scatter(void)
{

  long long ntot, ntotleft;
  int *noffset, *nbuffer, *nsend, *nsend_local, *numlist, *ndonelist;
  int i, j, n, ndone, maxfill, source, iter = 0;
  int level, ngrp, sendTask, recvTask, place, nexport;
  double tstart, tend, tstart_ngb = 0, tend_ngb = 0;
  double sumt, sumcomm, timengb, sumtimengb;
  double timecomp = 0, timeimbalance = 0, timecommsumm = 0, sumimbalance;
  MPI_Status status;

  // MPUEL: add lists for numbers of annihilations/scatterings and oscillation parameters
#ifdef OSCILLATION_DM
  double cosmictime, previouscosmictime;
  if(All.ComovingIntegrationOn)
  {
    cosmictime = get_cosmic_time(All.Time); // for all particles is the same
    if(All.PreviousTime != 0)
      previouscosmictime = get_cosmic_time(All.PreviousTime);
    else
      previouscosmictime = 0;
  }
  else
  {
    cosmictime = All.Time;
    previouscosmictime = All.PreviousTime;
  }
#endif

#ifdef ANNIHILATION_DM
  unsigned long *N_annihilationsList;
#endif
#ifdef SCATTERING_DM
  unsigned long *N_scatteringsList;
#endif

#ifdef PERIODIC // important because it defines the boxSize
  boxSize = All.BoxSize;
  boxHalf = 0.5 * All.BoxSize;
#ifdef LONG_X
  boxHalf_X = boxHalf * LONG_X;
  boxSize_X = boxSize * LONG_X;
#endif
#ifdef LONG_Y
  boxHalf_Y = boxHalf * LONG_Y;
  boxSize_Y = boxSize * LONG_Y;
#endif
#ifdef LONG_Z
  boxHalf_Z = boxHalf * LONG_Z;
  boxSize_Z = boxSize * LONG_Z;
#endif
#endif // end PERIODIC

  noffset = malloc(sizeof(int) * NTask);  /* offsets of bunches in common list */
  nbuffer = malloc(sizeof(int) * NTask);
  nsend_local = malloc(sizeof(int) * NTask);
  nsend = malloc(sizeof(int) * NTask * NTask);
  ndonelist = malloc(sizeof(int) * NTask);

// MPUEL: set up space for lists for numbers of annihilations/scatterings
#ifdef ANNIHILATION_DM
  N_annihilationsList = malloc(sizeof(unsigned long) * NTask);
#endif
#ifdef SCATTERING_DM
  N_scatteringsList = malloc(sizeof(unsigned long) * NTask);
#endif


  for(n = 0, NumADMUpdate = 0; n < NumPart; n++) // MPUEL: modified this loop
  {
    if(P[n].Type == ADM_type)
    {

#ifdef OSCILLATION_DM
      P[n].OscillationAngle += All.delta_m * (cosmictime - previouscosmictime);
      
      if(ThisTask == 0 && n==0)
      {
        printf("All.Time = %f\n", All.Time);
        printf("cosmictime = %f\n", cosmictime);
        printf("previouscosmictime = %f\n", previouscosmictime);
        printf("\nP[%d].OscillationAngle = %f\n", n, P[n].OscillationAngle);
      }
      if(ThisTask == 0 && n==1)
      {
        printf("P[%d].OscillationAngle = %f\n", n, P[n].OscillationAngle);
      }
#endif

      if(P[n].Ti_begstep == All.Ti_Current) // if(P[n].Ti_endstep == All.Ti_Current) is wrong because the Ti_*step is already advanced in advance_and_find_timesteps() in "run.c" but All.Ti_Current is changed only after in find_next_sync_point_and_drift() in "run.c"
        NumADMUpdate++;
    }
  }

  numlist = malloc(NTask * sizeof(int) * NTask);
  MPI_Allgather(&NumADMUpdate, 1, MPI_INT, numlist, 1, MPI_INT, MPI_COMM_WORLD);
  for(i = 0, ntot = 0; i < NTask; i++)
    ntot += numlist[i]; // it counts all the ADM active particles in the simulation
  free(numlist);



  /* we will repeat the whole thing for those particles where we didn't
   * find enough neighbours (it is useful just for varaible search radii, 
   * but not in the constant case as we consider)
   */
  do // it just does one cycle!
  {
    i = 0;      /* begin with this index */
    ntotleft = ntot;    /* ADM particles left for all tasks together */

    while(ntotleft > 0)
    {
      
#ifdef SCATTERING_DM
      All.N_scatterings = 0; // initialize N_scatterings 
#endif
#ifdef ANNIHILATION_DM
      All.N_annihilations = 0; // initialize N_annihilations
#endif

      for(j = 0; j < NTask; j++)
        nsend_local[j] = 0;

      //if (ThisTask == 0) // MPUEL
      //  if (i == 0)
      //    printf("******* Here we are!\n");

      /* do local particles and prepare export list */
      tstart = second();
      for(nexport = 0, ndone = 0; i < NumPart && nexport < All.BunchSizeADM - NTask; i++)
        if(P[i].Ti_begstep == All.Ti_Current) // same motivation as on line 319 // loop under all the particles in the local processor, NumPart
        {
          if(P[i].Type == ADM_type)
          {
            ndone++;

            for(j = 0; j < NTask; j++)
              Exportflag[j] = 0;

            do_annihilation_scattering(i, 0); // Exportflag is updated in ngb_treefind_variable_ADM() function

            for(j = 0; j < NTask; j++)
            {
              if(Exportflag[j]) // if there are pseudo-particles, namely they reside in the communication buffer
              {
                ADMDataIn[nexport].Pos[0] = P[i].Pos[0];
                ADMDataIn[nexport].Pos[1] = P[i].Pos[1];
                ADMDataIn[nexport].Pos[2] = P[i].Pos[2];
                ADMDataIn[nexport].Vel[0] = P[i].Vel[0];
                ADMDataIn[nexport].Vel[1] = P[i].Vel[1];
                ADMDataIn[nexport].Vel[2] = P[i].Vel[2];
                ADMDataIn[nexport].Index = i;
                ADMDataIn[nexport].Task = j; // needed for adm_compare_key()
                ADMDataIn[nexport].Ti_begstep = P[i].Ti_begstep;
                ADMDataIn[nexport].Ti_endstep = P[i].Ti_endstep;
                ADMDataIn[nexport].Core2 = ThisTask;
#ifdef OSCILLATION_DM
                ADMDataIn[nexport].OscillationAngle = P[i].OscillationAngle;
#endif
#ifdef ANNIHILATION_DM
                ADMDataIn[nexport].AnnihilateFlag = P[i].AnnihilateFlag;
#endif
#ifdef SCATTERING_DM
                ADMDataIn[nexport].dt_scatter = P[i].dt_scatter;
                ADMDataIn[nexport].ScatterFlag = P[i].ScatterFlag;
#endif
                nexport++;
                nsend_local[j]++;
              }
            }
          }
        }

      tend = second();
      timecomp += timediff(tstart, tend);

      qsort(ADMDataIn, nexport, sizeof(struct admdata_in), adm_compare_key); // sort the particles based on which processer (Task) they are

      for(j = 1, noffset[0] = 0; j < NTask; j++)
        noffset[j] = noffset[j - 1] + nsend_local[j - 1];

      tstart = second();

      MPI_Allgather(nsend_local, NTask, MPI_INT, nsend, NTask, MPI_INT, MPI_COMM_WORLD);

      tend = second();
      timeimbalance += timediff(tstart, tend);


      /* now do the particles that need to be exported */

      for(level = 1; level < (1 << PTask); level++)
      {
        tstart = second();
        for(j = 0; j < NTask; j++)
          nbuffer[j] = 0;
        for(ngrp = level; ngrp < (1 << PTask); ngrp++)
        {
          maxfill = 0;
          for(j = 0; j < NTask; j++)
          {
            if((j ^ ngrp) < NTask)
              if(maxfill < nbuffer[j] + nsend[(j ^ ngrp) * NTask + j])
                maxfill = nbuffer[j] + nsend[(j ^ ngrp) * NTask + j];
          }
          if(maxfill >= All.BunchSizeADM)
            break;

          sendTask = ThisTask;
          recvTask = ThisTask ^ ngrp;

          if(recvTask < NTask)
          {
            if(nsend[ThisTask * NTask + recvTask] > 0 || nsend[recvTask * NTask + ThisTask] > 0)
            {
              /* get the particles */
              MPI_Sendrecv(&ADMDataIn[noffset[recvTask]],
                nsend_local[recvTask] * sizeof(struct admdata_in), MPI_BYTE,
                recvTask, TAG_ADM_A,
                &ADMDataGet[nbuffer[ThisTask]],
                nsend[recvTask * NTask + ThisTask] * sizeof(struct admdata_in),
                MPI_BYTE, recvTask, TAG_ADM_A, MPI_COMM_WORLD, &status);
            } // here we pass from ADMDataIn to ADMDataGet
          }

          for(j = 0; j < NTask; j++)
            if((j ^ ngrp) < NTask)
              nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];
        }

        tend = second();
        timecommsumm += timediff(tstart, tend);


        tstart = second();

        /* Here we scatter imported particles */

        for(j = 0; j < nbuffer[ThisTask]; j++)
          do_annihilation_scattering(j, 1); // here we pass from ADMDataGet to ADMDataResult
          
        tend = second();
        timecomp += timediff(tstart, tend);

        /* do a block to explicitly measure imbalance */
        tstart = second();
        MPI_Barrier(MPI_COMM_WORLD);
        tend = second();
        timeimbalance += timediff(tstart, tend);

        /* get the result */
        tstart = second();
        for(j = 0; j < NTask; j++)
          nbuffer[j] = 0;
        
        for(ngrp = level; ngrp < (1 << PTask); ngrp++)
        {
          maxfill = 0;
          for(j = 0; j < NTask; j++)
          {
            if((j ^ ngrp) < NTask)
              if(maxfill < nbuffer[j] + nsend[(j ^ ngrp) * NTask + j])
                maxfill = nbuffer[j] + nsend[(j ^ ngrp) * NTask + j];
          }
          if(maxfill >= All.BunchSizeADM)
            break;

          sendTask = ThisTask;
          recvTask = ThisTask ^ ngrp;

          if(recvTask < NTask)
          {
            if(nsend[ThisTask * NTask + recvTask] > 0 || nsend[recvTask * NTask + ThisTask] > 0)
            {
              /* send the results */
              MPI_Sendrecv(&ADMDataResult[nbuffer[ThisTask]],
                nsend[recvTask * NTask + ThisTask] * sizeof(struct admdata_out),
                MPI_BYTE, recvTask, TAG_ADM_B,
                &ADMDataPartialResult[noffset[recvTask]],
                nsend_local[recvTask] * sizeof(struct admdata_out),
                MPI_BYTE, recvTask, TAG_ADM_B, MPI_COMM_WORLD, &status);

              /* add the result to the particles */
              for(j = 0; j < nsend_local[recvTask]; j++)
              {
                source = j + noffset[recvTask];
                place = ADMDataIn[source].Index;

                P[place].Vel[0] += ADMDataPartialResult[source].DeltaVel[0];
                P[place].Vel[1] += ADMDataPartialResult[source].DeltaVel[1];
                P[place].Vel[2] += ADMDataPartialResult[source].DeltaVel[2];

#ifdef OSCILLATION_DM
                P[place].OscillationAngle += ADMDataPartialResult[source].DeltaOscillationAngle;
#endif

#ifdef ANNIHILATION_DM
                P[place].AnnihilateFlag += ADMDataPartialResult[source].DeltaAnnihilateFlag;
#endif
#ifdef SCATTERING_DM
                P[place].dt_scatter     += ADMDataPartialResult[source].Deltadt_scatter;
                P[place].ScatterFlag    += ADMDataPartialResult[source].DeltaScatterFlag;
#endif
              }
            }
          }

          for(j = 0; j < NTask; j++)
            if((j ^ ngrp) < NTask)
              nbuffer[j] += nsend[(j ^ ngrp) * NTask + j];
        }
          
        tend = second();
        timecommsumm += timediff(tstart, tend);

        level = ngrp - 1;
      } // end for-loop starting from line 365

      MPI_Allgather(&ndone, 1, MPI_INT, ndonelist, 1, MPI_INT, MPI_COMM_WORLD);
      for(j = 0; j < NTask; j++)
        ntotleft -= ndonelist[j];


// MPUEL: Sum all the number of annihilations and/or scatterings (inspired by Rocha's work)
#ifdef ANNIHILATION_DM
      MPI_Allgather(&All.N_annihilations, 1, MPI_UNSIGNED_LONG, N_annihilationsList, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
      if(ThisTask == 0) // done by the local processor
      {
        for(i = 0; i < NTask; i++)
          All.N_annihilations_in_timestep += N_annihilationsList[i];
      }
#endif
#ifdef SCATTERING_DM
      MPI_Allgather(&All.N_scatterings, 1, MPI_UNSIGNED_LONG, N_scatteringsList, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
      if(ThisTask == 0)
      {
        for(i = 0; i < NTask; i++)
          All.N_scatterings_in_timestep += N_scatteringsList[i];
      }
#endif


    } // closes while-loop starting from line 334


    if(ntot > 0)
    {
      if(iter == 0)
        tstart_ngb = second();

      iter++;

      if(iter > 0 && ThisTask == 0)
      {
        printf("ngb iteration %d: need to repeat for %d%09d particles.\n", iter,
          (int) (ntot / 1000000000), (int) (ntot % 1000000000));
        fflush(stdout);
      }

      if(iter > MAXITER)
      {
        printf("failed to converge in neighbour iteration in density()\n");
        fflush(stdout);
        endrun(1155);
      }
    }
    else
      tend_ngb = second();

    /* when not using variable search radii */
    ntot = 0; // therefore iter = 1

  }
  while(ntot > 0); // close do-loop from line 329


  free(ndonelist);
  free(nsend);
  free(nsend_local);
  free(nbuffer);
  free(noffset);

// MPUEL: free the lists
#ifdef ANNIHILATION_DM
  free(N_annihilationsList);
  All.N_annihilations_tot += All.N_annihilations_in_timestep;
#endif
#ifdef SCATTERING_DM
  free(N_scatteringsList);
  All.N_scatterings_tot += All.N_scatterings_in_timestep;
#endif


#if defined(ANNIHILATION_DM) && (!defined(TEST_HERNQUIST_HALO_ANN))
  remove_annihilated_particles(); // remove DM particles whose AnnihilateFlag == 1
#endif

#ifdef SCATTERING_DM
  update_timesteps_after_scattering(); // update the timesteps for particles having new dt_scatter
#endif


  // MPUEL: since iter = 1 because the while(ntot > 0) part does just one cycle, we need to define tend_ngb
  if(ntot == 0)
    tstart_ngb = second();


  /* collect some timing information */
  if(iter > 0)
    timengb = timediff(tstart_ngb, tend_ngb);
  else
    timengb = 0;

  MPI_Reduce(&timengb, &sumtimengb, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timecomp, &sumt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timecommsumm, &sumcomm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timeimbalance, &sumimbalance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(ThisTask == 0)
  {
    /* there CPU names are not used in GADGET-2 (except _EnsureNgb one) 
     * because there are no gas particles, so in FdCPU file, the corresponding item
     * is due not to Hydro computations, but to ADM model */
    All.CPU_HydCompWalk += sumt / NTask;
    All.CPU_HydCommSumm += sumcomm / NTask;
    All.CPU_HydImbalance += sumimbalance / NTask;
    All.CPU_EnsureNgb += sumtimengb / NTask;
  }


}



/*! Inspired by density_evaluate() in density.c
 *  This function represents the core of the annihilation and scattering processes.
 *  The target particle may either be local, or reside in the communication
 *  buffer.
 */
void do_annihilation_scattering(int target, int mode)
{
  int j, k, n;
  int startnode, numngb_inbox = 0;
  double dt_fac;
  double dx, dy, dz, r2 = 0;
  double dvx, dvy, dvz, v, v2;
  double h, h2;
  //short int isAnnihilationProcess = 0; // if 0, scattering can occur; if 1, annihilation can occur // OLD CHOICE
  double original_vel[3];
  FLOAT *pos, *vel;

  // MPUEL: ADM varaibles
  int core2;
  double a = All.Time;
  double tmp = 0, dt = 0;

#ifdef OSCILLATION_DM
  double angosc = 0; // oscillation angle
  double original_OscillationAngle;
  double cosmic_time; // cosmic time
#endif

#ifdef ANNIHILATION_DM
  double h_ann, h2_ann, hinv_ann, hinv3_ann, hinv4_ann;
  double inv_vol_ann;
  double sig_ann_fac;
  int nannihilate = 0;
  double annihilate_rate, annihilate_prob, rand_annihilate;
  short int original_AnnihilateFlag, annflag;
#if defined(VECTOR_MEDIATOR) || defined(SCALAR_MEDIATOR)
  double epsV, epsA, arg1, arg2, arg3, arg4;
#endif
#endif // end ANNIHILATION_DM

#ifdef SCATTERING_DM
#ifndef TEST_HERNQUIST_HALO_SCATT
  FLOAT *vel_fin;
#endif
  double h_scatt, h2_scatt, hinv_scatt, hinv3_scatt, hinv4_scatt;
  double inv_vol_scatt;
  double sig_scatt_fac;
  double scatter_rate, scatter_prob;
  short int original_ScatterFlag, scattflag;
  int nscatter = 0;
  double w;
  double original_dt_scatter, dt_delimiter;
#if defined(VECTOR_MEDIATOR) || defined(SCALAR_MEDIATOR)
  double vrel, omega;
#ifndef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION
  double coeff = 0.01; // ratio between vrel and omega below which the vrel->0 limit cross section is taken
#endif
#endif // end VECTOR_MEDIATOR || SCALAR_MEDIATOR
#ifdef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION
  int vi = 0;
#endif
#endif // end SCATTERING_DM


  /* Convert cross-sections to physical units, and convert dlog(a) to dt */
  /* Note: GADGET internal units are: 
   * - positions are in comoving coordinates -> physical ones are obtained by multiplying by (a/h)
   * - velocities are the canonical momenta  -> physical ones are obtained by multiplying by (1/a)
   * - masses contain a h dependence         -> physical ones are obtained by multiplying by (1/h)
   */

  if(All.ComovingIntegrationOn)
  {
      dt_fac = 1.0 / (All.Hubble*All.HubbleParam * sqrt(All.Omega0 / (a * a * a) + (1 - All.Omega0 - All.OmegaLambda) / (a * a) + All.OmegaLambda));

#ifdef OSCILLATION_DM
      cosmic_time = get_cosmic_time(All.Time);
#endif

#ifdef ANNIHILATION_DM // s-wave
      sig_ann_fac = All.HubbleParam*All.HubbleParam / (a*a*a); // (h/a)^3 for the inv_vol; (1/h) for particle mass m inside sigma/m (m_p is already in correct units)
#endif
#ifdef SCATTERING_DM
      sig_scatt_fac = All.HubbleParam*All.HubbleParam / (a*a*a*a); // (h/a)^3 for the inv_vol; (1/a) for velocity; (1/h) for particle mass m inside sigma/m (m_p is already in correct units)
#endif

  }
  else
  {
      dt_fac = 1.0;

#ifdef OSCILLATION_DM
      cosmic_time = All.Time;
#endif

#ifdef ANNIHILATION_DM
      sig_ann_fac = 1.0;
#endif
#ifdef SCATTERING_DM
      sig_scatt_fac = 1.0;
#endif

  }


  if(mode == 0) /* This is a local particle */
  {
    pos = P[target].Pos;
    vel = P[target].Vel;

    tmp = P[target].Ti_endstep - P[target].Ti_begstep;
    if(tmp > 0)
      dt = All.Timebase_interval * tmp;
    tmp = 0;

    core2 = ThisTask;

#ifdef OSCILLATION_DM
    angosc = P[target].OscillationAngle;
#endif

#ifdef ANNIHILATION_DM
    annflag = P[target].AnnihilateFlag;
#endif

#ifdef SCATTERING_DM
    dt_delimiter = P[target].dt_scatter;
    scattflag = P[target].ScatterFlag;
#endif

  }
  else
  {
    pos = ADMDataGet[target].Pos;
    vel = ADMDataGet[target].Vel;

    tmp = ADMDataGet[target].Ti_endstep - ADMDataGet[target].Ti_begstep;
    if(tmp > 0)
      dt = All.Timebase_interval * tmp;
    tmp = 0;

    core2 = ADMDataGet[target].Core2;

    for(k = 0; k < 3; k++)
      original_vel[k] = ADMDataGet[target].Vel[k];

#ifdef OSCILLATION_DM
    angosc = ADMDataGet[target].OscillationAngle;

    original_OscillationAngle = ADMDataGet[target].OscillationAngle;
#endif // end OSCILLATION_DM

#ifdef ANNIHILATION_DM
    annflag = ADMDataGet[target].AnnihilateFlag;

    original_AnnihilateFlag = ADMDataGet[target].AnnihilateFlag;
#endif

#ifdef SCATTERING_DM
    dt_delimiter = ADMDataGet[target].dt_scatter;
    scattflag = ADMDataGet[target].ScatterFlag;

    original_dt_scatter = ADMDataGet[target].dt_scatter;
    original_ScatterFlag = ADMDataGet[target].ScatterFlag;
#endif

  }



#ifdef ANNIHILATION_DM
  // we can put it here since h_scatt is fixed for all particles
  h_ann = All.AnnihilateSearchRadius; 
  if(All.Time * All.AnnihilateSearchRadius > All.AnnihilateSearchRadiusMaxPhys)   
    h_ann = All.AnnihilateSearchRadiusMaxPhys / All.Time; // AnnihilateSearchRadius does not contain All.Time dependence

  h2_ann = h_ann * h_ann;
  hinv_ann = 1.0 / h_ann;
#ifndef  TWODIMS
  hinv3_ann = hinv_ann * hinv_ann * hinv_ann;
#else
  hinv3_ann = hinv_ann * hinv_ann / boxSize_Z;
#endif
  hinv4_ann = hinv3_ann * hinv_ann;

  inv_vol_ann = hinv3_ann / (4.0 * M_PI / 3.0);  //4.18879020479 is (4/3)pi

#ifndef SCATTERING_DM // not defined!
  h = h_ann;
  h2 = h2_ann;
#endif
#endif // end ANNIHILATION_DM


#ifdef SCATTERING_DM
  // we can put it here since h_scatt is fixed for all particles
  h_scatt = All.ScatterSearchRadius; 
  if(All.Time * All.ScatterSearchRadius > All.ScatterSearchRadiusMaxPhys)   
    h_scatt = All.ScatterSearchRadiusMaxPhys / All.Time; // ScatteringSearchRadius does not contain All.Time dependence

  h2_scatt = h_scatt * h_scatt;
  hinv_scatt = 1.0 / h_scatt;
#ifndef  TWODIMS
  hinv3_scatt = hinv_scatt * hinv_scatt * hinv_scatt;
#else
  hinv3_scatt = hinv_scatt * hinv_scatt / boxSize_Z;
#endif
  hinv4_scatt = hinv3_scatt * hinv_scatt;

  inv_vol_scatt = hinv3_scatt / (4.0 * M_PI / 3.0);  //4.18879020479 is (4/3)pi

#ifndef ANNIHILATION_DM // not defined!
  h = h_scatt;
  h2 = h2_scatt;
#endif
#endif // end SCATTERING_DM


#if defined(ANNIHILATION_DM) && defined(SCATTERING_DM)
  if(h_ann > h_scatt)
  {
    h = h_ann;
    h2 = h2_ann;
  }
  else
  {
    h = h_scatt;
    h2 = h2_scatt;
  }
#endif


  startnode = All.MaxPart; // it starts from the last particle
  numngb_inbox = 0;

  do
  {
    // h is the largest among h_ann and h_scatt
    numngb_inbox = ngb_treefind_variable_ADM(pos, h, &startnode); // it is equivalent to replace pos <-> &pos[0]

    for(n = 0; n < numngb_inbox; n++)
    {
      
      j = Ngblist[n];

      dx = pos[0] - P[j].Pos[0];
      dy = pos[1] - P[j].Pos[1];
      dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC     /*  now find the closest image in the given box size  */
      if(dx > boxHalf_X)
        dx -= boxSize_X;
      if(dx < -boxHalf_X)
        dx += boxSize_X;
      if(dy > boxHalf_Y)
        dy -= boxSize_Y;
      if(dy < -boxHalf_Y)
        dy += boxSize_Y;
      if(dz > boxHalf_Z)
        dz -= boxSize_Z;
      if(dz < -boxHalf_Z)
        dz += boxSize_Z;
#endif
      r2 = dx * dx + dy * dy + dz * dz;

      // h2 is the largest among h2_ann and h2_scatt
      if(r2 < h2) // within search radius, so there is chance to annihilate or scatter!
      {

        dvx = vel[0] - P[j].Vel[0];
        dvy = vel[1] - P[j].Vel[1];
        dvz = vel[2] - P[j].Vel[2];
                    
        v2 = dvx * dvx + dvy * dvy + dvz * dvz;
        v = sqrt(v2);


#ifdef ANNIHILATION_DM

        // check if the target particle has already annihilated
        if(annflag == 1)
          break; //exit the for loop

        // check if neigbor particle has already annihilated
        if(P[j].AnnihilateFlag == 1)
          continue;


        if(r2 < h2_ann)
        {

          annihilate_rate = All.sigmav_s * inv_vol_ann * P[j].Mass * sig_ann_fac;


#if defined(VECTOR_MEDIATOR) || defined(SCALAR_MEDIATOR)
          // add Sommerfeld enhancement factor
          epsV = (v/a) / (2.0 * All.VectorOrScalarCoupling) / All.c; // (1/a) in order to get the physical velocity, then divide by the speed of light to get dimensionless quantity
          epsA = All.MediatorDMmassratio / All.VectorOrScalarCoupling; // dimensionless

          arg1 = 12.0 * epsV / (M_PI * epsA);
          arg3 = 6.0 / (pow(M_PI, 2) * epsA);
          arg4 = 36.0 * pow(epsV, 2) / (pow(M_PI, 4) * pow(epsA, 2));
          if((arg3 - arg4) >= 0)
            arg2 = 2.0 * M_PI * sqrt(arg3 - arg4);
          else
            arg2 = 2.0 * M_PI * sqrt(arg3);

          tmp = M_PI / epsV * sinh(arg1) / (cosh(arg1) - cos(arg2));

          if(tmp < 1) tmp = 1; // the Sommerfeld enhancement should be greater than 1

          annihilate_rate *= tmp;
#endif // end VECTOR_MEDIATOR || SCALAR_MEDIATOR


#if defined(OSCILLATION_DM) && defined(VECTOR_MEDIATOR)
          tmp = sin(angosc - P[j].OscillationAngle);
          annihilate_rate *= (tmp * tmp);
          tmp = 0;
#endif // end OSCILLATION_DM && VECTOR_MEDIATOR
// for SCALAR_MEDIATOR, see below


          if (v2 == 0 || r2 == 0)
            annihilate_rate = 0; // when the particle distance is zero or the relative velocity is zero (avoid patological events)

          annihilate_prob = annihilate_rate * dt * dt_fac;
        
        }
        else
          annihilate_prob = 0; // in the case r2 > h2_ann


        //if (ThisTask == 0)
        //if (annihilate_prob != 0) //  if (target == 0)
        //    printf("\nP[%d]: distance = %f, annihilate_rate = %f, annihilate_prob = %f\n", target, sqrt(r2), annihilate_rate, annihilate_prob);

#endif // ANNIHILATION_DM



#ifdef SCATTERING_DM

#ifdef TEST_UNIFORM_BKG // allows just one scattering
        // check if the target particle has already scattered
        if(scattflag == 1)
          break; //exit the for loop

        // check if neighbor particle has already scattered
        if(P[j].ScatterFlag == 1)
          continue;
#endif // end TEST_UNIFORM_BKG


        if(r2 < h2_scatt)
        {

          scatter_rate = All.sigma_scatter * v * inv_vol_scatt * P[j].Mass * sig_scatt_fac;


// Here we implement the (modified) scattering CROSS_SECTION options!
#if defined(VECTOR_MEDIATOR) || defined(SCALAR_MEDIATOR)

          vrel = (v/a) / All.c; // (1/a) in order to get the physical velocity, then divide by the speed of light to get dimensionless quantity
          // we can do that, because only the ratio between v and All.MediatorDMmassratio enters the formulae below
          omega = All.MediatorDMmassratio;

#ifdef TRANSFER_CROSS_SECTION
#if !defined(MODIFIED_TRANSFER_CROSS_SECTION) && !defined(VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
          // transfer cross section
          tmp = 2.0 * pow(omega, 4) / pow(vrel, 4) * ( log(1.0 + pow(vrel, 2) / pow(omega, 2)) - pow(vrel, 2) / (pow(vrel, 2) + pow(omega, 2)) );
          if(vrel < coeff * omega)
            tmp = 1.0 - 4.0 * pow(vrel, 2) / (3.0 * pow(omega, 2)); // vrel-> 0 limit
          if(tmp >= 0)
            scatter_rate *= tmp;
          else
            scatter_rate = 0;
          tmp = 0;
#endif
#endif // end TRANSFER_CROSS_SECTION

#ifdef MODIFIED_TRANSFER_CROSS_SECTION
#if !defined(TRANSFER_CROSS_SECTION) && !defined(VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION)
          // modified transfer cross section
          // we multiply by (2.0) to make it equal to sigma in the isotropic regime
          tmp = (2.0)* 2.0 * pow(omega, 4) / pow(vrel, 4) * ( 2.0 * log(1.0 + pow(vrel, 2) / (2.0 * pow(omega, 2))) - log(1.0 + pow(vrel, 2) / pow(omega, 2)) );
          if(vrel < coeff * omega)
            tmp = (2.0)* 0.5 * (1.0 - pow(vrel, 2) / pow(omega, 2)); // vrel-> 0 limit
          if(tmp >= 0)
            scatter_rate *= tmp;
          else
            scatter_rate = 0;
          tmp = 0;
#endif
#endif // end MODIFIED_TRANSFER_CROSS_SECTION

#if !defined(TRANSFER_CROSS_SECTION) && !defined(MODIFIED_TRANSFER_CROSS_SECTION)
#ifndef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION 
          // viscosity cross section
          // we multiply by 3.0/2.0 to make it equal to sigma in the isotropic regime
          tmp = (3.0/2.0)* 4.0 * pow(omega, 4) / pow(vrel, 4) * ( (1.0 + 2.0 * pow(omega, 2) / pow(vrel, 2)) * log(1.0 + pow(vrel, 2) / pow(omega, 2)) - 2.0 );
          if(vrel < 3.0 * coeff * omega)
            tmp = (3.0/2.0)* 2.0 / 3.0 * (1.0 - pow(vrel, 2) / pow(omega, 2)); // vrel-> 0 limit
          if(tmp >= 0)
            scatter_rate *= tmp;
          else
            scatter_rate = 0;
          tmp = 0;
#endif
#endif

#endif // end VECTOR_MEDIATOR || SCALAR_MEDIATOR


#ifdef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION
          /* index of this velocity in scattering tables */
          vi = (All.Nv - 1)*log10(v/All.vmin)/log10(All.vmax/All.vmin);

          if (vi < 0) vi = 0;

          if (vi > All.Nv-1) scatter_rate = 0;
          else scatter_rate *= All.CrossSectionTable[vi];        
#endif


//#ifdef SCATTERING_DM_ONLY_ALONG_ZAXIS // for test 1: scattering in a uniform background
//          if (dvx * dvx + dvy * dvy > 0) scatter_rate = 0; // scatterig just along the z-axis (it occurs just once in principle for out test)         
//#endif


          scatter_prob = scatter_rate * dt * dt_fac;


        }
        else
          scatter_prob = 0; // in the case r2 > h2_scatt


        //if (ThisTask == 0)
        //if (scatter_prob != 0) //  if (target == 0)
        //    printf("\nP[%d]: distance = %f, scatter_rate = %f, scatter_prob = %f\n", target, sqrt(r2), scatter_rate, scatter_prob);

#endif // end SCATTERING_DM



        /* avoid double counting */
                    
        if (ThisTask == core2)
        {

#ifdef ANNIHILATION_DM
          annihilate_prob *= 0.5;
#endif
#ifdef SCATTERING_DM
          scatter_prob *= 0.5;
#endif

        }
        else // Robertson's idea: one-way send/receive between pairs of cores
        {
          if ( (ThisTask+core2) % 2 == 0 )
          {
            if (ThisTask < core2)
            {

#ifdef ANNIHILATION_DM
              annihilate_prob = 0;
#endif
#ifdef SCATTERING_DM
              scatter_prob = 0;
#endif
            }
          }
          else
          {   
            if (ThisTask > core2)
            {

#ifdef ANNIHILATION_DM
              annihilate_prob = 0;
#endif
#ifdef SCATTERING_DM
              scatter_prob = 0;
#endif
            }
          }
        }


        /* TEST */
        /* if (ThisTask == 0)
            if (target == 0)
                printf("\n\n\n\n\n Found neighbour at relative pos [%f,%f,%f] h^-1 Mpc \nRelative velocity \%f km/s \nScatter_rate = %e \nScatter_prob = %e n\n\n\n\n ", dx,dy,dz, v, scatter_rate, scatter_prob); */
        /* TEST */


#ifdef ANNIHILATION_DM

        rand_annihilate = gsl_rng_uniform(random_generator_ADM);


#if defined(OSCILLATION_DM) && defined(SCALAR_MEDIATOR)

        // check whether the particles would have annihilated without modulated factor
        if (rand_annihilate < annihilate_prob) // would have annihilated!
        {
          angosc *= exp(-1.0);
          P[j].OscillationAngle *= exp(-1.0);
        }

        tmp = sin(angosc + P[j].OscillationAngle);
        annihilate_prob *= (tmp * tmp);
        tmp = 0;

#endif // end OSCILLATION_DM && SCALAR_MEDIATOR


        // check for annihilation (if annihilate_prob == 0, the following does not occur)
        if (rand_annihilate < annihilate_prob) // annihilate!
        {

          //if (ThisTask == 0)
          //  printf("--> P[%d] and P[%d] have annihilated!   annihilate_prob = %f\n", target, j, annihilate_prob);

#ifndef TEST_HERNQUIST_HALO_ANN
          annflag = 1;
          P[j].AnnihilateFlag = 1;
#else
          fprintf(FdAdmModel, "%f   %f   %f   %f   %f   %f   %f\n", pos[0], pos[1], pos[2], P[j].Pos[0], P[j].Pos[1], P[j].Pos[2], All.Time);
#endif

          nannihilate ++;

          break; // once the target particle is annihilated, exit the for-loop

        }

#endif // end ANNIHILATION_DM



#ifdef SCATTERING_DM

        // check for scattering (if scatter_prob == 0, the following does not occur)
        if (gsl_rng_uniform(random_generator_ADM) < scatter_prob) // scatter!
        {

          //if (ThisTask == 0)
          //  printf("--> P[%d] and P[%d] have scattered!   scatter_prob = %f\n", target, j, scatter_prob);

          
          if (scatter_prob > All.ProbabilityTol)
          {
            dt_delimiter = All.ProbabilityTol / (scatter_rate * dt_fac);
            P[j].dt_scatter = dt_delimiter; // Assumption: DM particles and antiparticle have all equal mass! (correct: the Majorana mass is effectively negligible)
          }


          scattflag = 1;
          P[j].ScatterFlag = 1;


          // Assumption: DM particles and antiparticles have all equal mass!  (correct: the Majorana mass is effectively negligible)
          w = 0.5 * v; // v is the relative velocity


#ifndef TEST_HERNQUIST_HALO_SCATT
#ifndef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION // if not defined! --> isotropic scattering is more common

          // isotropic scattering                            
          vel_fin = isotropic_velocities(vel, P[j].Vel, w);

          // updated, so future scatterings will have correct vel
          vel[0] = vel_fin[0];
          vel[1] = vel_fin[1];
          vel[2] = vel_fin[2];
                            
          // local particle, can just update velocity
          P[j].Vel[0] = vel_fin[3];
          P[j].Vel[1] = vel_fin[4];
          P[j].Vel[2] = vel_fin[5];

#else
          // Angular-dependent scattering
          vel_fin = anisotropic_velocities(vel, P[j].Vel, w, dvx, dvy, dvz, vi);
                            
          vel[0] = vel_fin[0];
          vel[1] = vel_fin[1];
          vel[2] = vel_fin[2];
                            
          P[j].Vel[0] = vel_fin[3];
          P[j].Vel[1] = vel_fin[4];
          P[j].Vel[2] = vel_fin[5];
                            
#endif // end VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION
#endif // end not TEST_HERNQUIST_HALO_SCATT


#if defined(OSCILLATION_DM) && defined(VECTOR_MEDIATOR)

#ifdef LATE_TIME_PHASE_CHANGE // only for vector mediator because decoherence causes phase-change and therefore annihilation
          if(cosmic_time >= 1.0 / All.delta_m)
          {
            angosc = fmod(angosc, (2.0 * M_PI)) * exp(-3.0/2.0); // we might change it in the future, to simply take order(1) change in the phase after each scattering
            P[j].OscillationAngle = fmod(P[j].OscillationAngle, (2.0 * M_PI)) * exp(-3.0/2.0); // we might change it in the future, to simply take order(1) change in the phase after each scattering
          }
#else // standard case (= if not LATE_TIME_PHASE_CHANGE)
          angosc = fmod(angosc, (2.0 * M_PI)) * exp(-3.0/2.0);
          P[j].OscillationAngle = fmod(P[j].OscillationAngle, (2.0 * M_PI)) * exp(-3.0/2.0);
#endif

#endif // end defined(OSCILLATION_DM) && defined(VECTOR_MEDIATOR)


#ifdef TEST_HERNQUIST_HALO_SCATT
          fprintf(FdAdmModel, "%f   %f   %f   %f   %f   %f   %f\n", pos[0], pos[1], pos[2], P[j].Pos[0], P[j].Pos[1], P[j].Pos[2], All.Time);
#endif
          
          nscatter ++;

        }

#endif // end SCATTERING_DM
        

      } // closes if-condition starting from line 925
    } // closes for-loop starting from line 898

#ifdef ANNIHILATION_DM
    if(annflag == 1)
      break; // if the target particle has been annihilated, exit the while-loop too
#endif

#if defined(SCATTERING_DM) && defined(TEST_UNIFORM_BKG) // allows just one scattering
        if(scattflag == 1)
          break; // if the target particle has already scattered, exit the while-loop too
#endif // end TEST_UNIFORM_BKG

  }
  while(startnode >= 0); // closes do-while loop starting from line 894



#ifdef ANNIHILATION_DM
  All.N_annihilations += nannihilate;
#endif
#ifdef SCATTERING_DM
  All.N_scatterings += nscatter;
#endif


  if(mode == 0)
  {
    for(k = 0; k < 3; k++)
      P[target].Vel[k] = vel[k];

#ifdef OSCILLATION_DM
    P[target].OscillationAngle = angosc;
#endif

#ifdef ANNIHILATION_DM
    P[target].AnnihilateFlag = annflag;
#endif
#ifdef SCATTERING_DM
    P[target].dt_scatter = dt_delimiter;
    P[target].ScatterFlag = scattflag;
#endif

  }
  else
  {
    for(k = 0; k < 3; k++)
      ADMDataResult[target].DeltaVel[k] = vel[k] - original_vel[k]; // valid for both scattering and annihilation!

#ifdef OSCILLATION_DM
      ADMDataResult[target].DeltaOscillationAngle = angosc - original_OscillationAngle;
#endif

#ifdef ANNIHILATION_DM
      ADMDataResult[target].DeltaAnnihilateFlag = annflag - original_AnnihilateFlag;
#endif
#ifdef SCATTERING_DM
      ADMDataResult[target].Deltadt_scatter = dt_delimiter - original_dt_scatter;
      ADMDataResult[target].DeltaScatterFlag = scattflag - original_ScatterFlag;
#endif

  }
}



/*! Copied from ngb.c, and altered slightly for ADM_type particle search.
 *  This function returns neighbours with distance <= hsml and returns them in
 *  Ngblist. Actually, particles in a box of half side length hsml are
 *  returned, i.e. the reduction to a sphere still needs to be done in the
 *  calling routine. 
 *  The distinction between DM particle and antiparticle is done in the calling routine
 *  which is in do_annihilation_scattering() function.
 */
int ngb_treefind_variable_ADM(FLOAT searchcenter[3], FLOAT hsml, int *startnode)
{
  int k, numngb;
  int no, p;
  struct NODE *this;
  FLOAT searchmin[3], searchmax[3];

#ifdef PERIODIC
  double xtmp;
#endif

  for(k = 0; k < 3; k++)  /* cube-box window */
  {
    searchmin[k] = searchcenter[k] - hsml;
    searchmax[k] = searchcenter[k] + hsml;
  }

  numngb = 0;
  no = *startnode;

  while(no >= 0)
  {
    if(no < All.MaxPart)  /* single particle */
    {
      p = no;
      no = Nextnode[no];

      if(P[p].Type != ADM_type) // select only ADM_type particles because otherwise it continues in the while loop
        continue;

#ifdef PERIODIC
      if(NGB_PERIODIC_X(P[p].Pos[0] - searchcenter[0]) < -hsml)
        continue;
      if(NGB_PERIODIC_X(P[p].Pos[0] - searchcenter[0]) > hsml)
        continue;
      if(NGB_PERIODIC_Y(P[p].Pos[1] - searchcenter[1]) < -hsml)
        continue;
      if(NGB_PERIODIC_Y(P[p].Pos[1] - searchcenter[1]) > hsml)
        continue;
      if(NGB_PERIODIC_Z(P[p].Pos[2] - searchcenter[2]) < -hsml)
        continue;
      if(NGB_PERIODIC_Z(P[p].Pos[2] - searchcenter[2]) > hsml)
        continue;
#else
      if(P[p].Pos[0] < searchmin[0])
        continue;
      if(P[p].Pos[0] > searchmax[0])
        continue;
      if(P[p].Pos[1] < searchmin[1])
        continue;
      if(P[p].Pos[1] > searchmax[1])
        continue;
      if(P[p].Pos[2] < searchmin[2])
        continue;
      if(P[p].Pos[2] > searchmax[2])
        continue;
#endif // end PERIODIC

      Ngblist[numngb++] = p;

      if(numngb == MAX_NGB)
      {
        numngb = ngb_clear_buf(searchcenter, hsml, numngb);
        if(numngb == MAX_NGB)
        {
          printf("ThisTask=%d: Need to do a second neighbour loop for (%g|%g|%g) hsml=%g no=%d\n",
            ThisTask, searchcenter[0], searchcenter[1], searchcenter[2], hsml, no);
          *startnode = no;
          return numngb;
        }
      }
    }
    else
    {
      if(no >= All.MaxPart + MaxNodes)  /* pseudo particle, namely resides in the communication buffer */
      {
        Exportflag[DomainTask[no - (All.MaxPart + MaxNodes)]] = 1;
        no = Nextnode[no - MaxNodes];
        continue;
      }

      this = &Nodes[no];

      no = this->u.d.sibling; /* in case the node can be discarded */
#ifdef PERIODIC
      if((NGB_PERIODIC_X(this->center[0] - searchcenter[0]) + 0.5 * this->len) < -hsml)
        continue;
      if((NGB_PERIODIC_X(this->center[0] - searchcenter[0]) - 0.5 * this->len) > hsml)
        continue;
      if((NGB_PERIODIC_Y(this->center[1] - searchcenter[1]) + 0.5 * this->len) < -hsml)
        continue;
      if((NGB_PERIODIC_Y(this->center[1] - searchcenter[1]) - 0.5 * this->len) > hsml)
        continue;
      if((NGB_PERIODIC_Z(this->center[2] - searchcenter[2]) + 0.5 * this->len) < -hsml)
        continue;
      if((NGB_PERIODIC_Z(this->center[2] - searchcenter[2]) - 0.5 * this->len) > hsml)
        continue;
#else
      if((this->center[0] + 0.5 * this->len) < (searchmin[0]))
        continue;
      if((this->center[0] - 0.5 * this->len) > (searchmax[0]))
        continue;
      if((this->center[1] + 0.5 * this->len) < (searchmin[1]))
        continue;
      if((this->center[1] - 0.5 * this->len) > (searchmax[1]))
        continue;
      if((this->center[2] + 0.5 * this->len) < (searchmin[2]))
        continue;
      if((this->center[2] - 0.5 * this->len) > (searchmax[2]))
        continue;
#endif // end PERIODIC

      no = this->u.d.nextnode;  /* ok, we need to open the node */
    }
  }

  *startnode = -1;
  return numngb;
}



// MPUEL: inspired by Robertson's work
FLOAT *isotropic_velocities(FLOAT p1vel[3], FLOAT p2vel[3], double w0)
{
  double phi, cos_phi, sin_phi, cos_theta, sin_theta;
  double CoM_x, CoM_y, CoM_z;

  static FLOAT p1p2vel[6];


  phi = 2.0 * M_PI * gsl_rng_uniform(random_generator_ADM);
  cos_phi = cos(phi);
  sin_phi = sin(phi);
                            
  cos_theta = 1 - 2 * gsl_rng_uniform(random_generator_ADM);
  sin_theta = sqrt(1 - pow(cos_theta, 2));
                            
  CoM_x = 0.5 * (p1vel[0] + p2vel[0]);
  CoM_y = 0.5 * (p1vel[1] + p2vel[1]);
  CoM_z = 0.5 * (p1vel[2] + p2vel[2]);
                            
  p1p2vel[0] = CoM_x + w0 * cos_theta;
  p1p2vel[1] = CoM_y + w0 * sin_theta * cos_phi;
  p1p2vel[2] = CoM_z + w0 * sin_theta * sin_phi;
                            
  p1p2vel[3] = CoM_x - w0 * cos_theta;
  p1p2vel[4] = CoM_y - w0 * sin_theta * cos_phi;
  p1p2vel[5] = CoM_z - w0 * sin_theta * sin_phi;


  return p1p2vel;
}


#ifdef VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION

// MPUEL: inspired by Robertson's work
FLOAT *anisotropic_velocities(FLOAT p1vel[3], FLOAT p2vel[3], double w0, double dvx0, double dvy0, double dvz0, int vi0)
{
  double phi, cos_phi, sin_phi, theta, cos_theta, sin_theta;
  double CoM_x, CoM_y, CoM_z;

  double w1[3];
  float ex[3], ey[3], ez[3];
  float x1[3];
  int d;
  double norm;
  int Z;

  static FLOAT p1p2vel[6];


  // set up unit vectors in CoM frame
  w1[0] = 0.5 * dvx0;
  w1[1] = 0.5 * dvy0;
  w1[2] = 0.5 * dvz0;
                            
  for(d = 0; d < 3; d++) ez[d] = w1[d] / w0;
                            
  if (fabsf(ez[0]) < 0.707106781) // check ez not (1,0,0) // 0.7 = sin(45) = cos(45)
  {   
    ex[0] = 0;
    ex[1] = ez[2];
    ex[2] = -ez[1];
  }
  else
  {   
    ex[0] = -ez[2];
    ex[1] = 0;
    ex[2] = ez[0];
  }
                            
  norm = 1.0 / sqrt(ex[0]*ex[0] + ex[1]*ex[1] + ex[2]*ex[2]);
                            
  for(d = 0; d < 3; d++) ex[d] *= norm;
                            
  ey[0] = ez[1]*ex[2] - ez[2]*ex[1];
  ey[1] = ez[2]*ex[0] - ez[0]*ex[2];
  ey[2] = ez[0]*ex[1] - ez[1]*ex[0];
                            
  norm = 1.0 / sqrt(ey[0]*ey[0] + ey[1]*ey[1] + ey[2]*ey[2]);
                            
  for(d = 0; d < 3; d++)
    ey[d] *= norm; // now we have unit vectors
                            
                            
  phi = 2.0 * M_PI * gsl_rng_uniform(random_generator_ADM);
                            
  Z = All.Ntheta * gsl_rng_uniform(random_generator_ADM);
  theta = All.ThetaTable[vi0 * All.Ntheta + Z];


#ifdef SCATTERING_DM_RELABEL                            
  if(theta > (M_PI/2.0)) theta = M_PI - theta;
#endif // end SCATTERING_DM_RELABEL


  cos_theta = cos(theta);
  sin_theta = sin(theta);
  cos_phi = cos(phi);
  sin_phi = sin(phi);
                            
  // set x1 in lab orientation
  for(d = 0; d < 3; d++) 
    x1[d] = w0 * sin_theta * cos_phi * ex[d] + w0 * sin_theta * sin_phi * ey[d] + w0 * cos_theta * ez[d];
                            
                            
  CoM_x = 0.5 * (p1vel[0] + p2vel[0]);
  CoM_y = 0.5 * (p1vel[1] + p2vel[1]);
  CoM_z = 0.5 * (p1vel[2] + p2vel[2]);
                            
  p1p2vel[0] = CoM_x + x1[0];
  p1p2vel[1] = CoM_y + x1[1];
  p1p2vel[2] = CoM_z + x1[2];
                            
  p1p2vel[3] = CoM_x - x1[0];
  p1p2vel[4] = CoM_y - x1[1];
  p1p2vel[5] = CoM_z - x1[2];  


  return p1p2vel;  
}

#endif //end VELOCITY_ANGULAR_DEPENDENT_CROSS_SECTION



#ifdef ANNIHILATION_DM

// MPUEL: remove annihilated DM particles
void remove_annihilated_particles(void)
{

  int i, k;
  double t0, t1;

  t0 = second();

  for(i = 0; i < NumPart; i++)
  {
    if(P[i].Type == ADM_type && P[i].AnnihilateFlag == 1) // only new annihilated particles
    {

      P[i].Mass = 0; // we assign zero mass because we do not want it to contribute to the gravitational force computatation
      P[i].Type = annihilated_type; // we assign different type to reduce the number of ngb searches and reduce the size of snapshot file
      
      P[i].Ti_begstep = TIMEBASE - 1;
      P[i].Ti_endstep = TIMEBASE; // end of the time interval of the simulation

      for(k = 0; k < 3; k++) 
      {
        P[i].Vel[k] = 0;
        P[i].GravAccel[k] = 0;
#ifdef PMGRID
        P[i].GravPM[k] = 0;
#endif
#ifdef FORCETEST
        P[i].GravAccelDirect[k] = 0;
#endif
      }

      P[i].Potential = 0;
      P[i].OldAcc = 0;

#ifdef FLEXSTEPS
      P[i].FlexStepGrp = 0;
#endif

      P[i].GravCost = 0; // since it should not affect gravitational computation

#ifdef PSEUDOSYMMETRIC
      P[i].AphysOld = 0;
#endif      

    }
  }

  t1 = second();
  All.CPU_TimeLine += timediff(t0, t1);
}

#endif // ANNIHILATION_DM



#ifdef SCATTERING_DM

// MPUEL: update the timesteps of particles that scatter if they are below their current timesteps
/*! Inspired by advance_and_find_timesteps() and get_timestep() in timestep.c.
 *  It updates the timesteps for the ADM_type particles that scattered
 *  and whose dt_scatter is below the timestep derived from the kick function
 *  advance_and_find_timesteps(), which occurs before annhilate_scatter() in run.c
 */
void update_timesteps_after_scattering(void)
{
  int i, ti_step_adm, ti_min;
  double t0, t1;

#ifndef NOSTOP_WHEN_BELOW_MINTIMESTEP
  double fac1, ax, ay, az, ac;
#endif

#ifdef FLEXSTEPS
  int ti_grp;
#endif

  t0 = second();

  for(i = 0; i < NumPart; i++)
  {

    if(P[i].Type == ADM_type && P[i].dt_scatter > 0)
    {

      if(P[i].dt_scatter < All.MinSizeTimestep) // dt_scatter is already in dt or dlog(a), as All.MinSizeTimestep is
      {

#ifndef NOSTOP_WHEN_BELOW_MINTIMESTEP        
        if(All.ComovingIntegrationOn)
          fac1 = 1 / (All.Time * All.Time);
        else
          fac1 = 1;

        ax = fac1 * P[i].GravAccel[0];
        ay = fac1 * P[i].GravAccel[1];
        az = fac1 * P[i].GravAccel[2];

#ifdef PMGRID
        ax += fac1 * P[i].GravPM[0];
        ay += fac1 * P[i].GravPM[1];
        az += fac1 * P[i].GravPM[2];
#endif

        ac = sqrt(ax * ax + ay * ay + az * az); /* this is the physical acceleration */

        if(ac == 0)
          ac = 1.0e-30;

        printf("warning: Timestep wants to be below the limit `MinSizeTimestep'\n");
        printf("Part-ID=%d  dt=%g ac=%g xyz=(%g|%g|%g)\n", (int) P[i].ID, P[i].dt_scatter, ac, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);

        fflush(stdout);
        endrun(888);
#endif // end NOSTOP_WHEN_BELOW_MINTIMESTEP
        
        P[i].dt_scatter = All.MinSizeTimestep; // P[i].dt_scatter can not be below All.MinSizeTimestep!
      }

      ti_step_adm = P[i].dt_scatter / All.Timebase_interval;


#ifdef FLEXSTEPS
      if((All.Ti_Current % (4 * All.PresentMinStep)) == 0)
        if(All.PresentMinStep < TIMEBASE)
          All.PresentMinStep *= 2;


      /* make it a power 2 subdivision */
      ti_min = TIMEBASE;
      while(ti_min > ti_step_adm)
        ti_min >>= 1; // divide by 2 repeatedly
      ti_step_adm = ti_min;

      if(ti_step_adm < All.PresentMinStep)
        All.PresentMinStep = ti_step_adm;


      ti_step_adm = All.PresentMinStep;
      MPI_Allreduce(&ti_step_adm, &All.PresentMinStep, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

      All.PresentMaxStep = ti_min;

      if(ThisTask == 0)
        printf("Syn Range = %g  PresentMinStep = %d  PresentMaxStep = %d \n",
          (double) All.PresentMaxStep / All.PresentMinStep, All.PresentMinStep, All.PresentMaxStep);

#endif // end FLEXSTEPS


      /* make it a power 2 subdivision */
      ti_min = TIMEBASE;
      while(ti_min > ti_step_adm)
        ti_min >>= 1; // divide by 2 repeatedly
      ti_step_adm = ti_min;


#ifdef FLEXSTEPS
      ti_grp = P[i].FlexStepGrp % All.PresentMaxStep;
      ti_grp = (ti_grp / All.PresentMinStep) * All.PresentMinStep;
      ti_step_adm = ((P[i].Ti_begstep + ti_grp + ti_step_adm) / ti_step_adm) * ti_step_adm - (P[i].Ti_begstep + ti_grp);
#endif


      if(All.Ti_Current == TIMEBASE)  /* we here finish the last timestep. */
        ti_step_adm = 0;

      if((TIMEBASE - All.Ti_Current) < ti_step_adm) /* check that we don't run beyond the end */
        ti_step_adm = TIMEBASE - All.Ti_Current;

    // if(P[i].Ti_begstep == All.Ti_Current) // not done because not only active particles might have different dt_scatter (since also scattered particles change it!)
      if(ti_step_adm < (P[i].Ti_endstep - P[i].Ti_begstep)) // if ti_step_adm is smaller than the previous timestep of ADM particle i
      {
        if(ti_step_adm != 0)
          printf("ADM modified timestep for Part-ID=%d: dt_scatter=%g\n", (int) P[i].ID, P[i].dt_scatter);
        
        // P[i].Ti_begstep remains the same, it is only the P[i].Ti_endstep that is reduced!
        P[i].Ti_endstep = P[i].Ti_begstep + ti_step_adm; // which is smaller than the previous P[i].Ti_endstep
      }
    
    }

    if (P[i].Type == ADM_type)
      P[i].dt_scatter = 0; // reset to initial value for next scattering (it is a dynamical varaible)

  }

  t1 = second();
  All.CPU_TimeLine += timediff(t0, t1);
}

#endif // end SCATTERING_DM


// MPUEL: new

/*! This routine is a comparison kernel used in a sort routine to group
 *  particles that are exported to the same processor.
 */
int adm_compare_key(const void *a, const void *b)
{
  if(((struct admdata_in *) a)->Task < (((struct admdata_in *) b)->Task))
    return -1;

  if(((struct admdata_in *) a)->Task > (((struct admdata_in *) b)->Task))
    return +1;

  return 0;
}



#endif // end defined(ANNIHILATION_DM) || defined(SCATTERING_DM)

#endif // end ADM_MODEL_ON
