// I removed most things related to working on a fraction of the data set.
// Also gone are functions dealing with loading the text data and creating binary files.
#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <map>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <cblas.h>

using namespace std;

//==================================================================
//
// Constants and Type Declarations
//
//===================================================================
#define TRAINING_PATH   "/home/srh2144/netflix/download/training_set/"
#define TRAINING_FILE   "/home/srh2144/netflix/download/training_set/%s"
#define TEST_PATH       "/home/srh2144/netflix/calculate_score/qualifying.txt"
#define PREDICTION_FILE "Netflixprediction.txt"

#define MAX_RATINGS     100480508     // Ratings in entire training set (+1)
#define MAX_RATINGS2    100480508     // Ratings in entire training set (+1)
#define MAX_CUSTOMERS   480190        // Customers in the entire training set (+1)
#define MAX_MOVIES      17771         // Movies in the entire training set (+1)
#define MAX_FEATURES    100     // Number of features to use 
#define MAX_ITERATIONS  60		  // Max iterations of ALS
#define LAMBDA			.06			  // regularization parameter

typedef unsigned char BYTE;
typedef map<int, int> IdMap;
typedef IdMap::iterator IdItr;

struct Movie {
    int         RatingCount;
    int         RatingSum;
    double      RatingAvg;            
    double      PseudoAvg;            // Weighted average used to deal with small movie counts 
};

struct Customer {
    int         CustomerId;
    int         RatingCount;
    int         RatingSum;
};

struct Data {
    int         CustId;
    short       MovieId;
    BYTE        Rating;
};

bool CustomerLessThan(Data i, Data j) { return (i.CustId < j.CustId); }

void shuffle(int array[], int n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// LAPACK linear solver
extern "C" {
    static int dsysv(char UPLO, int N, int NRHS, double *A, int LDA, int *IPIV, double *B, int LDB,
            double *WORK, int LWORK) {

        extern void dsysv_(const char *UPLOp, const int *Np, const int *NRHSp, double *A, const int *LDAp,
                int *IPIV, double *B, const int *LDBp, double *WORK, const int *LWORKp, int *INFOp);
        int info;
        dsysv_(&UPLO, &N, &NRHS, A, &LDA, IPIV, B, &LDB, WORK, &LWORK, &info);
        return info;
    }
}

// I got lazy and made everything public, whatever
class Engine {
    //private:
    public:
        Data*           m_aRatings;                                     // Array of ratings data, sorted by movie
        Data*           m_aRatings2;                                    // Array of ratings data, sorted by user
        Movie           m_aMovies[MAX_MOVIES];                          // Array of movie metrics
        Customer        m_aCustomers[MAX_CUSTOMERS];                    // Array of customer metrics
        float           m_aMovieFeatures[MAX_MOVIES][MAX_FEATURES];     // Array of features by movie
        float           m_aCustFeatures[MAX_CUSTOMERS][MAX_FEATURES];   // Array of features by customer
        IdMap           m_mCustIds;                                     // Map for one time translation of ids to compact array index
        int				movieIndexStart[MAX_MOVIES];					// index in ratings array where each new movie starts
        int				customerIndexStart[MAX_CUSTOMERS];				// index in ratings2 array where each new customer starts

        inline double   PredictRating(short movieId, int custId);

        bool            ReadNumber(char* pwzBufferIn, int nLength, int &nPosition, char* pwzBufferOut);
        bool            ParseInt(char* pwzBuffer, int nLength, int &nPosition, int& nValue);

        int				lastCustomer;									// necessary when not loading all of the data
        int				lastMovie;
        int             myRatings;                                      // number of ratings in movie-sorted ratings array for particular processor
        int             myRatings2;                                     // same, but for customer-sorted ratings array
        int             myFirstCustomer;                                 
        int             myFirstMovie;                                    
        int				rank;
        int				numjobs;
        MPI_Datatype	mpi_Data, mpi_Movie, mpi_Cust;

        //public:
        Engine(int r, int n);
        ~Engine(void) { };

        void            CalcFeatures();
        void            UpdateUsers();
        void            UpdateMovies();
        void            ProcessTest(const char* pwzFile);
        void            LoadBinaries();
        void            LoadRatings(); // separately load the ratings binaries, since these are huge and distributed
        void			getDataTypes();
};

void broadcast(Engine* engine);

//===================================================================
//
// Program Main
//
//===================================================================
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    double start, mid, end;
    int rank, numjobs;

    MPI_Comm_size(MPI_COMM_WORLD, &numjobs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Engine* engine = new Engine(rank, numjobs);

    if (rank == 0) start = MPI_Wtime(); 
    engine->LoadBinaries();
    broadcast(engine);

    for (int i=0; i<numjobs; i++) {
        if (rank==i) engine->LoadRatings();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (rank == 0) { end = MPI_Wtime(); cout << "loading: " << end-start << " s" << endl; }

    engine->CalcFeatures();

    if (rank == 0) {
        engine->ProcessTest("qualifying.txt");
        mid = end; end = MPI_Wtime(); cout << "processing: " << end-mid << " s" << endl;
        cout << "Total: " << end-start << " s" << endl;
    }

    MPI_Finalize();

    return 0;
}

//===================================================================
//
// Engine Class 
//
//===================================================================

//-------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------

Engine::Engine(int r, int n) {
    lastMovie = MAX_MOVIES-1;
    lastCustomer = MAX_CUSTOMERS-1;
    rank = r; numjobs = n;
    getDataTypes();
}

//-------------------------------------------------------------------
// Calculations - This Paragraph contains all of the relevant code
//-------------------------------------------------------------------

void Engine::CalcFeatures() {
    int i, f, numCusts=lastCustomer/numjobs, numMovs=lastMovie/numjobs; 
    double start, end;

    // this stuff is for the 'remainder' movies/customers, since typically the number of processes won't evenly divide these
    // I tacked all the remainder onto the last process
    int* recvcountsCust, * displsCust; recvcountsCust = new int[numjobs]; displsCust = new int[numjobs];
    int* recvcountsMov, * displsMov; recvcountsMov = new int[numjobs]; displsMov = new int[numjobs];
    for (i=0; i<numjobs; i++) {
        recvcountsCust[i] = numCusts*MAX_FEATURES; recvcountsMov[i] = numMovs*MAX_FEATURES;
        displsCust[i] = i*numCusts*MAX_FEATURES; displsMov[i] = i*numMovs*MAX_FEATURES;
    }
    recvcountsCust[numjobs-1] += (lastCustomer%numjobs)*MAX_FEATURES; recvcountsMov[numjobs-1] += (lastMovie%numjobs)*MAX_FEATURES;

    // the main calculating loop
    if (rank==0) cout << "Doing ALS" << endl;
    for (i=0; i<MAX_ITERATIONS; i++) {
        start = MPI_Wtime(); if (rank==0) cout << "iteration: " << i+1 << endl;

        UpdateUsers(); if (rank==0) cout << "users updated ok" << endl;
        MPI_Allgatherv(&m_aCustFeatures[myFirstCustomer][0], numCusts*MAX_FEATURES, MPI_FLOAT, &m_aCustFeatures[1][0], recvcountsCust, displsCust, MPI_FLOAT, MPI_COMM_WORLD);

        UpdateMovies(); if (rank==0) cout << "movies updated ok" << endl;
        MPI_Allgatherv(&m_aMovieFeatures[myFirstMovie][0], numMovs*MAX_FEATURES, MPI_FLOAT, &m_aMovieFeatures[1][0], recvcountsMov, displsMov, MPI_FLOAT, MPI_COMM_WORLD);

        end = MPI_Wtime(); if (rank==0) cout << end-start << " s" << endl; 
    }
    delete [] recvcountsMov; delete [] displsMov; delete [] recvcountsCust; delete [] displsCust;
}

void Engine::UpdateUsers() {
    int cust, ratingCount;
    int i=0, f=0, r=0, lwork=1000, nrhs=1, info;
    const int n=MAX_FEATURES;
    char uplo = 'L';
    double A[n*n], u[n], V[n], work[lwork];
    double * Mi = NULL;
    int ipiv[n];

    while (r < myRatings2) {
        //first, an annoying loop to throw out bad data
        do {
            cust = m_aRatings2[r].CustId;
            ratingCount = (cust <= 0 || cust >= MAX_CUSTOMERS) ? MAX_MOVIES : m_aCustomers[cust].RatingCount;
            r++;
        }
        while ((ratingCount <= 0 || ratingCount >= MAX_MOVIES) && r < myRatings2);
        if (ratingCount <= 0 || ratingCount >= MAX_MOVIES || r >= myRatings2) break; 
        else r--;

        //initialize the various temporary arrays needed for the linear algebra problem
        try { Mi = new double [n*ratingCount]; }
        catch (bad_alloc&) { cout << "couldn't allocate memory for Mi" << endl; exit(1); }
        double * R = new double [ratingCount];
        for (i=0; i<n; i++) {
            u[i] = V[i] = 0;
            for (f=0; f<n; f++) {
                if (i==f) A[i*n + f] = 1;
                else A[i*n + f] = 0;
            }
        }
        for (i=0; cust == m_aRatings2[r].CustId && r < myRatings2; i++) {
            for (f=0; f<n; f++) Mi[f*ratingCount + i] = m_aMovieFeatures[m_aRatings2[r].MovieId][f];
            R[i] = double(m_aRatings2[r].Rating);
            r++;
        }
        //do the linear algebra
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, ratingCount, 1.0, Mi, ratingCount, LAMBDA*ratingCount, A, n);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, ratingCount, 1.0, Mi, ratingCount, R, 1, 0.0, V, 1);
        info = dsysv(uplo, n, nrhs, A, n, ipiv, V, n, work, lwork);

        //update the feature matrix
        for (f=0; f<n; f++) m_aCustFeatures[cust][f] = V[f];

        delete[] Mi; Mi = NULL;
        delete[] R;
    }
}

// virtually identical to UpdateCustomers
void Engine::UpdateMovies() {
    int mov, ratingCount;
    int i=0, f=0, r=0, lwork=1000, nrhs=1, info, n=MAX_FEATURES;
    char uplo = 'L';
    double A[n*n], u[n], V[n], work[lwork];
    double * Mi = NULL;
    int ipiv[n];

    while (r < myRatings) {
        do {
            mov = m_aRatings[r].MovieId;
            ratingCount = (mov <= 0 || mov >= MAX_MOVIES) ? MAX_CUSTOMERS : m_aMovies[mov].RatingCount;
            r++;
        }
        while ((ratingCount <= 0 || ratingCount >= MAX_CUSTOMERS) && r < myRatings);
        if (ratingCount <= 0 || ratingCount >= MAX_CUSTOMERS || r >= myRatings) break; // ignore bad data
        else r--;

        try { Mi = new double [n*ratingCount]; }
        catch (bad_alloc&) { cout << "couldn't allocate memory for Mi" << endl; exit(1); }
        double * R = new double [ratingCount];
        for (i=0; i<n; i++) {
            u[i] = V[i] = 0;
            for (f=0; f<n; f++) {
                if (i==f) A[i*n + f] = 1;
                else A[i*n + f] = 0;
            }
        }
        for (i=0; mov == m_aRatings[r].MovieId && r < myRatings; i++) {
            for (f=0; f<n; f++) Mi[f*ratingCount + i] = m_aCustFeatures[m_aRatings[r].CustId][f];
            R[i] = double(m_aRatings[r].Rating);
            r++;
        }

        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, ratingCount, 1.0, Mi, ratingCount, LAMBDA*ratingCount, A, n);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, ratingCount, 1.0, Mi, ratingCount, R, 1, 0.0, V, 1);
        info = dsysv(uplo, n, nrhs, A, n, ipiv, V, n, work, lwork);

        for (f=0; f<n; f++) m_aMovieFeatures[mov][f] = V[f];

        delete[] Mi; Mi = NULL;
        delete[] R;
    }
}

//
// PredictRating
// - This version is used for calculating the final results
// - It loops through the entire list of finished features
//
double Engine::PredictRating(short movieId, int custId) {
    double sum = 0;
    for (int f=0; f<MAX_FEATURES; f++) {
        sum += m_aMovieFeatures[movieId][f] * m_aCustFeatures[custId][f];
        if (sum > 5) sum = 5;
        if (sum < 1) sum = 1;
    }
    return sum;
}

//-------------------------------------------------------------------
// Data Loading / Saving
//-------------------------------------------------------------------

//
// ProcessTest
// - Load a sample set in the following format
//
//   <Movie1Id>:
//   <CustomerId>
//   <CustomerId>
//   ...
//   <Movie2Id>:
//   <CustomerId>
//
// - And write results:
//
//   <Movie1Id>:
//   <Rating>
//   <Raing>
//   ...
void Engine::ProcessTest(const char* pwzFile) {
    FILE *streamIn, *streamOut, *solution;
    char pwzBuffer[1000], pwzBuffer2[100];
    int custId, movieId, pos = 0, numValues = 0, actualRating;
    double rating, sumSquaredValues=0, delta;
    bool bMovieRow;

    sprintf(pwzBuffer, TEST_PATH, pwzFile);
    printf("Processing test: %s\n", TEST_PATH);

    streamIn = fopen(pwzBuffer, "r"); 
    if (streamIn==NULL) { cout << "File error, " << pwzFile << endl; exit(1);}
    streamOut = fopen(PREDICTION_FILE, "w");
    if (streamOut==NULL) { cout << "File error, " << PREDICTION_FILE << endl; exit(1);}
    solution = fopen("judging.txt", "r");
    if (solution==NULL) { cout << "File error, solution.txt" << endl; exit(1);}

    fgets(pwzBuffer, 1000, streamIn); fgets(pwzBuffer2, 100, solution);
    while (!feof(streamIn)) {
        bMovieRow = false;
        for (int i=0; i<(int)strlen(pwzBuffer); i++) bMovieRow |= (pwzBuffer[i] == 58); 
        pos = 0;
        if (bMovieRow) {
            ParseInt(pwzBuffer, (int)strlen(pwzBuffer), pos, movieId);
            // Write same row to results
            //            fputs(pwzBuffer,streamOut); 
        }
        else {
            ParseInt(pwzBuffer, (int)strlen(pwzBuffer), pos, custId);
            actualRating = atoi(&pwzBuffer2[0]);
            custId = m_mCustIds[custId];
            rating = PredictRating(movieId, custId);
            delta = rating - (double)actualRating;
            sumSquaredValues += delta*delta;
            numValues++;

            // Write predicted value
            //            sprintf(pwzBuffer,"%5.3f",rating);
            //            fputs(pwzBuffer,streamOut);
            //            fputs("\n",streamOut); 
        }
        fgets(pwzBuffer, 1000, streamIn); fgets(pwzBuffer2, 100, solution);
    }
    cout << "rmse: " << sqrt(sumSquaredValues/(double)numValues) << endl;

    // Cleanup
    fclose( streamIn );
    fclose( streamOut );
}

//-------------------------------------------------------------------
// Helper Functions
//-------------------------------------------------------------------
bool Engine::ReadNumber(char* pwzBufferIn, int nLength, int &nPosition, char* pwzBufferOut)
{
    int count = 0;
    int start = nPosition;
    char wc = 0;

    // Find start of number
    while (start < nLength)
    {
        wc = pwzBufferIn[start];
        if ((wc >= 48 && wc <= 57) || (wc == 45)) break;
        start++;
    }

    // Copy each character into the output buffer
    nPosition = start;
    while (nPosition < nLength && ((wc >= 48 && wc <= 57) || wc == 69  || wc == 101 || wc == 45 || wc == 46))
    {
        pwzBufferOut[count++] = wc;
        wc = pwzBufferIn[++nPosition];
    }

    // Null terminate and return
    pwzBufferOut[count] = 0;
    return (count > 0);
}

bool Engine::ParseInt(char* pwzBuffer, int nLength, int &nPosition, int& nValue)
{
    char pwzNumber[20];
    bool bResult = ReadNumber(pwzBuffer, nLength, nPosition, pwzNumber);
    nValue = (bResult) ? atoi(pwzNumber) : 0;
    return bResult;
}

void Engine::LoadRatings() {
    m_aRatings = new Data[myRatings];
    m_aRatings2 = new Data[myRatings2];

    int i=0, startRating = movieIndexStart[myFirstMovie], startRating2 = customerIndexStart[myFirstCustomer];

    FILE *ratingsFile;
    size_t result;
    // ratings array, sorted by movie
    ratingsFile = fopen("ratings.bin","rb");
    if (rank==0) cout << "loading ratings" << endl;
    if (ratingsFile==NULL) { cout << "File error, ratings.bin" << endl; exit(1);}
    fseek(ratingsFile, startRating*sizeof(Data), SEEK_SET);
    result = fread(m_aRatings, sizeof(Data), myRatings, ratingsFile);
    if (result != myRatings) { cout << "Reading error, ratings.bin" << endl; exit(3);}
    MPI_Barrier(MPI_COMM_WORLD);
    fclose(ratingsFile);

    // ratings2, sorted by customer instead of movie 
    ratingsFile = fopen("ratings2.bin","rb");
    if (rank==0) cout << "loading ratings2" << endl;
    if (ratingsFile==NULL) { cout << "File error, ratings2.bin"; exit(1);}
    fseek(ratingsFile, startRating2*sizeof(Data), SEEK_SET);
    result = fread(m_aRatings2, sizeof(Data), myRatings2, ratingsFile);
    if (result != myRatings2) { cout << "Reading error, ratings2.bin"; exit(3);}
    MPI_Barrier(MPI_COMM_WORLD);
    fclose(ratingsFile);
}

void Engine::LoadBinaries() {
    if (rank==0) {
        int i=0;
        FILE *moviesFile, *customersFile, *indexFile;
        size_t result;

        // index arrays, where in ratings arrays each new movie or customer starts
        cout << "loading indices" << endl;
        indexFile = fopen("movieIndex.bin","rb");
        if (indexFile==NULL) { cout << "File error, movieIndex.bin" << endl; exit(1);}
        result = fread(movieIndexStart, sizeof(int), MAX_MOVIES, indexFile);
        if (result != MAX_MOVIES) { cout << "Reading error, movieIndex.bin " << result << endl; exit(3);}
        fclose(indexFile);
        indexFile = fopen("customerIndex.bin","rb");
        if (indexFile==NULL) { cout << "File error, customerIndex.bin" << endl; exit(1);}
        result = fread(customerIndexStart, sizeof(int), MAX_CUSTOMERS, indexFile);
        if (result != MAX_CUSTOMERS) { cout << "Reading error, customerIndex.bin " << result << endl; exit(3);}
        fclose(indexFile);

        // movies array
        cout << "loading movies" << endl;
        moviesFile = fopen("movies.bin","rb");
        if (moviesFile==NULL) { cout << "File error, movies.bin" << endl; exit(1);}
        result = fread(m_aMovies, sizeof(Movie), MAX_MOVIES, moviesFile);
        if (result != MAX_MOVIES) { cout << "Reading error, movies.bin " << result << endl; exit(3);}
        fclose(moviesFile);
        // initialize movie features matrix with movie averages in first row, and small random numbers elsewhere
        for (int i=0; i<MAX_MOVIES; i++) {
            m_aMovieFeatures[i][0] = float(m_aMovies[i].RatingAvg);
            for (int f=1; f<MAX_FEATURES; f++) m_aMovieFeatures[i][f] = double(rand())/double(RAND_MAX)/5.0 - 0.1;
        }

        // customers array
        cout << "loading customers" << endl;
        customersFile = fopen("customers.bin","rb");
        if (customersFile==NULL) { cout << "File error, customers.bin" << endl; exit(1);}
        result = fread(m_aCustomers, sizeof(Customer), MAX_CUSTOMERS, customersFile);
        if (result != MAX_CUSTOMERS) { cout << "Reading error, customers.bin " << result << endl; exit(3);}
        fclose(customersFile);
        // fix customer ID map
        for (i=0;i<MAX_CUSTOMERS;i++) m_mCustIds[m_aCustomers[i].CustomerId] = i;
    }
} 

// create the custom data types needed for communication with MPI
void Engine::getDataTypes() {
    int blen[4] = {1, 1, 1, 1}, blen2[5] = {1, 1, 1, 1, 1};
    MPI_Aint indicesData[4] = {0, sizeof(int), sizeof(int)+sizeof(short), sizeof(Data)};
    MPI_Aint indicesMovie[5] = {0, sizeof(int), sizeof(int)+sizeof(int), sizeof(int)+sizeof(int)+sizeof(double), sizeof(Movie)};
    MPI_Aint indicesCust[4] = {0, sizeof(int), sizeof(int)+sizeof(int), sizeof(Customer)};
    MPI_Datatype oldtypesData[4] = {MPI_INT, MPI_SHORT, MPI_UNSIGNED_CHAR, MPI_UB};
    MPI_Datatype oldtypesMovie[5] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_UB};
    MPI_Datatype oldtypesCust[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_UB};
    MPI_Type_struct(4, blen, indicesData, oldtypesData, &mpi_Data);
    MPI_Type_struct(5, blen2, indicesMovie, oldtypesMovie, &mpi_Movie);
    MPI_Type_struct(4, blen, indicesCust, oldtypesCust, &mpi_Cust);
    MPI_Type_commit(&mpi_Data); MPI_Type_commit(&mpi_Movie); MPI_Type_commit(&mpi_Cust);
}

// broadcast the stuff the root process loaded by itself 
// also determine the first rating, and number of ratings for each processor
void broadcast(Engine* engine) {
    int rank = engine->rank, numjobs = engine->numjobs;
    int firstMov, firstCust, numMovies, numCustomers;
    MPI_Status stat; MPI_Request req;

    MPI_Bcast (&engine->movieIndexStart, MAX_MOVIES, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&engine->customerIndexStart, MAX_CUSTOMERS, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&engine->m_aMovieFeatures, MAX_FEATURES*MAX_MOVIES, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&engine->m_aCustFeatures, MAX_FEATURES*MAX_CUSTOMERS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&engine->m_aMovies, MAX_MOVIES, engine->mpi_Movie, 0, MPI_COMM_WORLD);
    MPI_Bcast (&engine->m_aCustomers, MAX_CUSTOMERS, engine->mpi_Cust, 0, MPI_COMM_WORLD);

    numCustomers = engine->lastCustomer/numjobs; numMovies = engine->lastMovie/numjobs;
    firstCust = rank*numCustomers+1; firstMov = rank*numMovies+1;

    if (rank==numjobs-1) engine->myRatings = MAX_RATINGS2-engine->movieIndexStart[firstMov];  
    else engine->myRatings = engine->movieIndexStart[firstMov+numMovies]-engine->movieIndexStart[firstMov];  
    if (rank==numjobs-1) engine->myRatings2 = MAX_RATINGS2-engine->customerIndexStart[firstCust];  
    else engine->myRatings2 = engine->customerIndexStart[firstCust+numCustomers]-engine->customerIndexStart[firstCust];  
    engine->myFirstCustomer = firstCust; engine->myFirstMovie = firstMov;
}
