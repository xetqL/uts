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
#include <float.h>

typedef double  GossipRawType;
typedef double* GossipRawTypePtr;

GossipRawTypePtr victimsData;
int MPIWS_GOSSIP_SHARE = 121231234;
int *victims;    // Size == worldsize

inline double* allocate_gossip_memory(int comm_size, int *buff_size) {
    *buff_size = 2 * comm_size * sizeof(GossipRawType);
    return (GossipRawTypePtr) malloc(2 * comm_size * sizeof(GossipRawType));
}

inline double get_victim_age(int rank) { return victimsData[2*rank+1]; }
inline double get_victim_wir(int rank) { return victimsData[2*rank];   }
inline void   set_victim_age(int rank, GossipRawType age) { victimsData[2*rank+1] = age; }
inline void   set_victim_wir(int rank, GossipRawType wir) { victimsData[2*rank]   = wir; }
inline double get_timestamp() { return MPI_Wtime(); }

inline void vsinit(int rank, int comm_size) {
    srand(rank);
    int i,j;

    // allocate & init the victim array
    victims     = (int*) malloc((comm_size -1) * sizeof(int));
    victimsData = (double*) malloc(2 * comm_size * sizeof(GossipRawType)); //first is wir, then age

    for(i = 0, j = 0; i < comm_size; i++) {
        if(i != rank) { victims[j++] = i; }
        set_victim_age(i, -1.0);
        set_victim_wir(i,  0.0);
    }

    if(rank == 0) {
        printf("set\n");
        set_victim_age(0,  get_timestamp());
        set_victim_wir(0,  18092013.0);
    }
}

inline char* vsdescript(void)
{
#if defined(__SS_HALF__)
    return "MPI Workstealing (Half Gossip Selection Candidate)";
#else
    return "MPI Workstealing (Gossip Selection Candidate)";
#endif
}

int seed_knowledge = 0;

#include <sys/time.h>

inline void gossip_merge_unpack(GossipRawTypePtr rcv_buff, int comm_size, int rank, int from)
{
    int i, position = 0;
    //merge with existing data
    for(i = 0; i < comm_size; ++i) {
        if(rcv_buff[2*i+1] > get_victim_age(i)) { // then replace data
            set_victim_wir(i, rcv_buff[2*i]);
            set_victim_age(i, rcv_buff[2*i+1]);
        }
    }
}

inline void gossip_pack(GossipRawTypePtr buffer, int comm_size){
    const int FSIZE = 2 * comm_size * sizeof(GossipRawType) ;
    memcpy(buffer, victimsData, FSIZE);
}

inline int selectvictim(int rank, int size, int last)
{

    int A, B, C;
    GossipRawType Awir, Bwir, Cwir;
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
    if(get_victim_age(A) == -1 && get_victim_age(B) == -1 && get_victim_age(C) == -1) {
        i = rand() % 3;
        switch(i)
        {
            case 0: return A;
            case 1: return B;
            case 2: return C;
        }
    }

    Awir = get_victim_wir(A);
    Bwir = get_victim_wir(B);
    Cwir = get_victim_wir(C);

    if(Awir > Bwir){
        if(Awir > Cwir)
            return A;
        else
            return C;
    } else if (Bwir > Cwir)
        return B;
    else {
        i = rand() % 3;
        switch(i)
        {
            case 0: return A;
            case 1: return B;
            default: return C;
        }
    }
}

#else
#error "You forgot to select a victim selection"
#endif /* Strategy selection */

#endif /* VICTIMSELECT_H */
