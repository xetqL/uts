#ifndef VICTIMSELECT_H
#define VICTIMSELECT_H 1

#if defined(__VS_ORIG__)

inline void vsinit(int rank, int size)
{
}

inline char * vsdescript(void)
{
#if defined(__SS_HALF__)
	return "MPI Workstealing (Half Round Robin)";
#else
	return "MPI Workstealing (Round Robin)";
#endif
}

inline int selectvictim(int rank, int size, int last)
{
	last = (last + 1) % size;
	if(last == rank) last = (last + 1) % size;
	return last;
}

#elif defined(__VS_RAND__)

inline void vsinit(int rank, int size)
{
	srand(rank);
}

inline char * vsdescript(void)
{
#if defined(__SS_HALF__)
	return "MPI Workstealing (Half Rand)";
#else
	return "MPI Workstealing (Rand)";
#endif
}

inline int selectvictim(int rank, int size, int last)
{
	do {
		last = rand()%size;
	} while(last == rank);
	return last;
}

#elif defined(__VS_GSLUI__)
#include <gsl/gsl_rng.h>
int *victims;
gsl_rng *rng;

inline char * vsdescript()
{
#if defined(__SS_HALF__)
	return "MPI Workstealing (Half GSL Uniform Int)";
#else
	return "MPI Workstealing (GSL Uniform Int)";
#endif
}

inline void vsinit(int rank, int size)
{
	int i,j;
	// allocate & init the victim array
	victims = malloc((size -1) * sizeof(int));
	for(i = 0, j = 0; i < size; i++)
		if(i != rank)
			victims[j++] = i;
	// init an rng, using the rank-th number from the global sequence as seed
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	rng = gsl_rng_alloc(T);
	for(i = 0; i < rank; i++)
		gsl_rng_uniform_int(rng,UINT_MAX);
	gsl_rng_set(rng,gsl_rng_uniform_int(rng,UINT_MAX));
}

inline int selectvictim(int rank, int size, int last)
{
	last = gsl_rng_uniform_int(rng,size -1);
	last = victims[last];
	return last;
}

#elif defined(__VS_GSLRD__)
#include <mpi-ext.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int *victims;
double *weights;
gsl_rng *rng;
gsl_ran_discrete_t *table;

inline void vsinit(int rank, int size)
{
	int i,j;
	// allocate & init the victim array
	victims = malloc((comm_size -1) * sizeof(int));
	for(i = 0, j = 0; i < comm_size; i++)
		if(i != comm_rank)
			victims[j++] = i;
	// compute weights, using the tofu coords
	int mx,my,mz,ma,mb,mc;
	FJMPI_Topology_sys_rank2xyzabc(comm_rank,&mx,&my,&mz,&ma,&mb,&mc);
	weights = malloc((comm_size -1) * sizeof(double));
	for(i = 0 ; i < comm_size-1; i++)
	{
		//find coords of this rank
		int x,y,z,a,b,c;
		FJMPI_Topology_sys_rank2xyzabc(victims[i],&x,&y,&z,&a,&b,&c);
		// compute euclidian distance between nodes
		double d = pow(mx - x,2) + pow(my - y,2) + pow(mz - z,2) + pow(ma - a,2) + pow(mb - b,2)
			+ pow(mc - c,2);
#if defined(__VS_FIX__)
		if(d < 1.0) d = 1.0;
#endif
		d = sqrt(d);
		weights[i] = 1.0/d;
	}
	// init an rng, using the rank-th number from the global sequence as seed
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	rng = gsl_rng_alloc(T);
	for(i = 0; i < rank; i++)
		gsl_rng_uniform_int(rng,UINT_MAX);
	gsl_rng_set(rng,gsl_rng_uniform_int(rng,UINT_MAX));
	table = gsl_ran_discrete_preproc(comm_size-1,weights);
}

inline char * vsdescript(void)
{
#if defined(__SS_HALF__)
	return "MPI Workstealing (Half GSL Distance Tofu)";
#else
	return "MPI Workstealing (GSL Tofu Weights)";
#endif
}

inline int selectvictim(int rank, int size, int last)
{
	last = gsl_ran_discrete(rng,table);
	last_steal = victims[last_steal];
	return last;
}

#elif defined(__GUIDED_WS__)
#include <mpi.h>
int *victims;    // Size == worldsize
double *victimsWir;
int *victimsAge;

inline char* allocate_gossip_memory(int comm_size, int *buff_size)
{
    *buff_size = comm_size * sizeof(double) + comm_size * sizeof(int);
    //printf("Buffer of size: %d\n", *buff_size);
    return (char*) malloc(*buff_size);
}

inline void vsinit(int rank, int comm_size)
{
    srand(rank);
    int i,j;
    // allocate & init the victim array
    victims    = malloc((comm_size -1) * sizeof(int));
    //weights = malloc((comm_size -1) * sizeof(double));
    victimsWir = calloc((comm_size)    , sizeof(double));
    victimsAge = malloc((comm_size)    * sizeof(int));

    for(i = 0, j = 0; i < comm_size; i++){
        if(i != rank) {
            victims[j++] = i;
        }
        victimsAge[i] = -1;
    }

}

inline char * vsdescript(void)
{
#if defined(__SS_HALF__)
    return "MPI Workstealing (Half Gossip Selection Candidate)";
#else
    return "MPI Workstealing (Gossip Selection Candidate)";
#endif
}

inline void gossip_merge_unpack(char* buffer, int comm_size)
{
    const int FSIZE = comm_size * sizeof(double) + comm_size * sizeof(int);
    int i, position = 0;

    //receiving buffer
    double *new_victimsWir = malloc(comm_size * sizeof(double));
    int    *new_victimsAge = malloc(comm_size * sizeof(int));

    //unpack buffer
    MPI_Unpack(buffer, FSIZE, &position, new_victimsWir, comm_size, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, FSIZE, &position, new_victimsAge, comm_size, MPI_INT,    MPI_COMM_WORLD);

    //merge with existing data
    for(i = 0; i < comm_size; ++i) {
        if(new_victimsAge[i] < victimsAge[i]){ //then replace data
            victimsWir[i] = new_victimsWir[i];
            victimsAge[i] = new_victimsAge[i];
        }
    }

    free(new_victimsWir);
    free(new_victimsAge);
}

inline void gossip_pack(char* buffer, int comm_size)
{
    const int FSIZE = comm_size * sizeof(double) + comm_size * sizeof(int);
    int i, position = 0;

    //Pack buffer
    MPI_Pack(victimsWir, comm_size, MPI_DOUBLE, buffer, FSIZE, &position, MPI_COMM_WORLD);
    MPI_Pack(victimsAge, comm_size, MPI_INT,    buffer, FSIZE, &position, MPI_COMM_WORLD);

    return;
}

inline int selectvictim(int rank, int size, int last)
{

    int A, B, C;
    double Awir, Bwir, Cwir;
    int i;

    do {
        A = rand() % size;
    } while(A == rank);

    do {
        B = rand() % size;
    } while(B == rank || B == A);

    do {
        C = rand() % size;
    } while(C == rank || C == B || C == A);

    //select randomly if age are the same
    if(victimsAge[A] == -1 && victimsAge[B] == -1 && victimsAge[C] == -1) {
        i = rand() % 3;
        switch(i)
        {
            case 0: return A;
            case 1: return B;
            case 2: return C;
        }
    }

    Awir = victimsWir[A];
    Bwir = victimsWir[B];
    Cwir = victimsWir[C];

    if(Awir >= Bwir && Awir >= Cwir) return A;
    if(Bwir >= Awir && Bwir >= Cwir) return B;
    if(Cwir >= Awir && Cwir >= Bwir) return C;
    return -1;
}

#else
#error "You forgot to select a victim selection"
#endif /* Strategy selection */

#endif /* VICTIMSELECT_H */
