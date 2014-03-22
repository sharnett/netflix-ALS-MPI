//#undef SEEK_SET
//#undef SEEK_END
//#undef SEEK_CUR
//#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <map>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include "/System/Library/Frameworks/vecLib.framework/Headers/cblas.h"

using namespace std;

//===================================================================
//
// Constants and Type Declarations
//
//===================================================================
#define TRAINING_PATH   "/Users/seanharnett/Desktop/netflix/download/training_set/"
#define TRAINING_FILE   "/Users/seanharnett/Desktop/netflix/download/training_set/%s"
#define TEST_PATH       "/Users/seanharnett/Desktop/netflix/calculate_score/qualifying.txt"
#define PREDICTION_FILE "Netflixprediction.txt"

#define MAX_RATINGS     100480508     // Ratings in entire training set (+1)
#define MAX_RATINGS2    100480508     // Ratings in entire training set (+1)
#define MAX_CUSTOMERS   480190        // Customers in the entire training set (+1)
#define MAX_MOVIES      17771         // Movies in the entire training set (+1)
#define MAX_FEATURES    5            // Number of features to use 
#define MIN_EPOCHS      5             // Minimum number of epochs per feature
#define MAX_EPOCHS      10            // Max epochs per feature
#define MAX_ITERATIONS  3			  // Max iterations of ALS

#define MIN_IMPROVEMENT 0.01          // Minimum improvement required to continue current feature
#define INIT            0.1           // Initialization value for features
#define LRATE           0.001         // Learning rate parameter
#define K               0.015         // Regularization parameter used to minimize over-fitting

typedef unsigned char BYTE;
typedef map<int, int> IdMap;
typedef IdMap::iterator IdItr;

struct Movie
{
    int         RatingCount;
    int         RatingSum;
    double      RatingAvg;            
    double      PseudoAvg;            // Weighted average used to deal with small movie counts 
};

struct Customer
{
    int         CustomerId;
    int         RatingCount;
    int         RatingSum;
};

struct Data
{
    int         CustId;
    short       MovieId;
    BYTE        Rating;
    float       Cache;
};

bool CustomerLessThan(Data i, Data j) { 
    return (i.CustId < j.CustId); 
}

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

class Engine 
{
private:
    int             m_nRatingCount;                                 // Current number of loaded ratings
    Data            m_aRatings[MAX_RATINGS2];                        // Array of ratings data, sorted by movie
    Data            m_aRatings2[MAX_RATINGS2];                       // Array of ratings data, sorted by user
    Movie           m_aMovies[MAX_MOVIES];                          // Array of movie metrics
    Customer        m_aCustomers[MAX_CUSTOMERS];                    // Array of customer metrics
    float           m_aMovieFeatures[MAX_FEATURES][MAX_MOVIES];     // Array of features by movie (using floats to save space)
    float           m_aCustFeatures[MAX_FEATURES][MAX_CUSTOMERS];   // Array of features by customer (using floats to save space)
    IdMap           m_mCustIds;                                     // Map for one time translation of ids to compact array index

    inline double   PredictRating(short movieId, int custId, int feature, float cache, bool bTrailing=true);
    inline double   PredictRating(short movieId, int custId);

    bool            ReadNumber(char* pwzBufferIn, int nLength, int &nPosition, char* pwzBufferOut);
    bool            ParseInt(char* pwzBuffer, int nLength, int &nPosition, int& nValue);
    bool            ParseFloat(char* pwzBuffer, int nLength, int &nPosition, float& fValue);

public:
    Engine(void);
    ~Engine(void) { };

    void            CalcMetrics();
    void            CalcFeatures2();
    void            UpdateUsers();
    void            UpdateMovies();
    void            LoadHistory();
    void            ProcessTest(const char* pwzFile);
    void            ProcessFile(char* pwzFile);
    void            CreateBinaries();
    void            LoadBinaries();
};


//===================================================================
//
// Program Main
//
//===================================================================
int main() {
    Engine* engine = new Engine();
	bool log = false;
    time_t start, mid, end;
    float loading, creating, metrics, features, processing, total;
    struct tm * timeinfo;

    time (&start); timeinfo = localtime(&start);
//    engine->LoadHistory();
    engine->LoadBinaries();
    time (&end); loading = difftime(end,start);
cout << "loading: " << loading << " s" << endl;

//	engine->CalcMetrics();
    mid = end; time (&end); metrics = difftime(end,mid);
cout << "metrics: " << metrics << " s" << endl;

//    engine->CreateBinaries();
    mid = end; time (&end); creating = difftime(end,mid);
cout << "creating: " << creating << " s" << endl;

	engine->CalcFeatures2();
    mid = end; time (&end); features = difftime(end,mid);

	engine->ProcessTest("qualifying.txt");
    mid = end; time (&end); processing = difftime(end,mid); total = difftime(end,start);
cout << "processing: " << processing << " s" << endl;

	if (log) {
		FILE *streamOut;
		if (!(streamOut = fopen("logfile.txt", "a"))) {
			cout << "failed to open logfile, aborting" << endl;
			return 0;
		}
		fprintf(streamOut, "%s", asctime(timeinfo));
		fprintf(streamOut, "MAX_FEATURES: %d\nMIN_EPOCHS: %d\nMAX_EPOCHS: %d\n", MAX_FEATURES, MIN_EPOCHS, MAX_EPOCHS);
		fprintf(streamOut, "MIN_IMPROVEMENT: %g\nINIT: %g\nLRATE: %g\nK: %g\n", MIN_IMPROVEMENT, INIT, LRATE, K);
		fprintf(streamOut, "Loading: %g\nMetrics: %g\nFeatures: %g\nProcessing: %g\nTotal: %g\n\n", loading, metrics, features, processing, total);
		fclose(streamOut);
	}
    cout << "Total: " << total << " s" << endl;

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

Engine::Engine(void) {
    m_nRatingCount = 0;
}

//-------------------------------------------------------------------
// Calculations - This Paragraph contains all of the relevant code
//-------------------------------------------------------------------

//
// CalcMetrics
// - Loop through the history and pre-calculate metrics used in the training 
// - Also re-number the customer id's to fit in a fixed array
//
void Engine::CalcMetrics()
{
    int i=0, cid;
    IdItr itr;
    printf("Calculating intermediate metrics\n");
    // Process each row in the training set
    for (i=0; i<m_nRatingCount; i++) {
        Data* rating = m_aRatings + i;

        // Increment movie stats
        m_aMovies[rating->MovieId].RatingCount++;
        m_aMovies[rating->MovieId].RatingSum += rating->Rating;
        
        // Add customers (using a map to re-number id's to array indexes) 
        itr = m_mCustIds.find(rating->CustId); 
        if (itr == m_mCustIds.end()) {
            cid = 1 + (int)m_mCustIds.size();

            // Reserve new id and add lookup
            m_mCustIds[rating->CustId] = cid;

            // Store off old sparse id for later
            m_aCustomers[cid].CustomerId = rating->CustId;

            // Init vars to zero
            m_aCustomers[cid].RatingCount = 0;
            m_aCustomers[cid].RatingSum = 0;
        }
        else {
            cid = itr->second;
        }

        // Swap sparse id for compact one
        rating->CustId = cid;

        m_aCustomers[rating->CustId].RatingCount++;
        m_aCustomers[rating->CustId].RatingSum += rating->Rating;
    }
    // Do a follow-up loop to calc movie averages
    for (i=0; i<MAX_MOVIES; i++) {
        Movie* movie = m_aMovies+i;
        movie->RatingAvg = movie->RatingSum / (1.0 * movie->RatingCount);
        movie->PseudoAvg = (3.23 * 25 + movie->RatingSum) / (25.0 + movie->RatingCount);
    }

//my special addition for ALS: start first row with average instead of .1
// also, other rows get small random numbers
    for (i=0; i<MAX_MOVIES; i++) {
		m_aMovieFeatures[0][i] = float(m_aMovies[i].RatingAvg);
		for (int f=1; f<MAX_FEATURES; f++) m_aMovieFeatures[f][i] = double(rand())/double(RAND_MAX)/5.0 - 0.1;
    }
    for (i=0; i<MAX_CUSTOMERS; i++) { 
		m_aCustFeatures[0][i] = float(m_aCustomers[i].RatingSum/(float(m_aCustomers[i].RatingCount)));
		for (int f=1; f<MAX_FEATURES; f++) m_aCustFeatures[f][i] = double(rand())/double(RAND_MAX)/5.0 - 0.1;
    }
//    for (i=0; i<m_nRatingCount; i++) m_aRatings2[i].CustId = m_mCustIds[m_aRatings2[i].CustId];

/*
ofstream boner;
boner.open("boners.txt");
boner << "m_aCustFeatures" << endl;
for (int f=0; f<MAX_FEATURES; f++) {
    for (i=10000; i<10020; i++) boner << setiosflags(ios::fixed) << setprecision(2) << setw(8) << m_aCustFeatures[f][i];
    boner << endl;
}
boner << "m_aMovieFeatures" << endl;
for (int f=0; f<MAX_FEATURES; f++) {
    for (i=10000; i<10020; i++) boner << setiosflags(ios::fixed) << setprecision(2) << setw(8) << m_aMovieFeatures[f][i];
    boner << endl;
}
//boner << "m_aCustomers" << endl;
//for (i=0; i<500; i++) boner << "i: " << i << " rating count: " << m_aCustomers[i].RatingCount <<  endl;
boner << "m_aRatings" << endl;
for (i=0; i<5000; i++)  {
	cid = m_aRatings[i].CustId;
	boner << "i: " << i << " cust: " << cid << " movie: " << m_aRatings[i].MovieId << " old cid: "  << m_aCustomers[cid].CustomerId << endl;
}
boner << "m_aRatings2" << endl;
for (i=0; i<5000; i++) {
	cid = m_aRatings2[i].CustId;
	boner << "i: " << i << " cust: " << cid << " movie: " << m_aRatings2[i].MovieId << " old cid: " << m_aCustomers[cid].CustomerId << endl;
}
boner.close();
*/
}

void Engine::CalcFeatures2() {
    time_t t0, t1;
    cout << "Doing ALS" << endl;
    for (int i=0; i<MAX_ITERATIONS; i++) {
	cout << "iteration: " << i+1 << endl;
	time(&t0);
	UpdateUsers();
    cout << "users updated ok" << endl;
	UpdateMovies();
    cout << "movies updated ok" << endl;
	time(&t1);
	cout << difftime(t1, t0) << " s" << endl; 
    }
}

//zzz
void Engine::UpdateMovies() {
    int mov, ratingCount;
    int i=0, f=0, r=1, lwork=1000, nrhs=1, info, n=MAX_FEATURES;
    char uplo = 'L';
    double A[n*n];
    double u[n];
    double V[n];
    double work[lwork];
    int ipiv[n];

    while (r < MAX_RATINGS2) {
        mov = m_aRatings[r].MovieId;
        ratingCount = m_aMovies[mov].RatingCount;
        double * Mi = new double [n*ratingCount];
        double * R = new double [ratingCount];
        for (i=0; i<n; i++) {
            u[i] = V[i] = 0;
            for (f=0; f<n; f++) {
                if (i==f) A[i*n + f] = 1;
                else A[i*n + f] = 0;
            }
        }
        for (i=0; mov == m_aRatings[r].MovieId && r < MAX_RATINGS2; i++) {
            for (f=0; f<n; f++) Mi[f*ratingCount + i] = m_aCustFeatures[f][m_aRatings[r].CustId];
            R[i] = double(m_aRatings[r].Rating);
            r++;
        }
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, ratingCount, 1.0, Mi, ratingCount, .05*ratingCount, A, n);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, ratingCount, 1.0, Mi, ratingCount, R, 1, 0.0, V, 1);
        info = dsysv(uplo, n, nrhs, A, n, ipiv, V, n, work, lwork);

        for (f=0; f<n; f++) m_aMovieFeatures[f][mov] = V[f];

        delete[] Mi;
        delete[] R;
    }
}

void Engine::UpdateUsers() {
    int cust, ratingCount;
    int i=0, f=0, r=1, lwork=1000, nrhs=1, info;
    const int n=MAX_FEATURES;
    char uplo = 'L';
    double A[n*n];
    double u[n];
    double V[n];
    double work[lwork];
    int ipiv[n];

    while (r < MAX_RATINGS2) {
        cust = m_aRatings2[r].CustId;
        ratingCount = m_aCustomers[cust].RatingCount;
        double * Mi = new double [n*ratingCount];
        double * R = new double [ratingCount];
        for (i=0; i<n; i++) {
            u[i] = V[i] = 0;
            for (f=0; f<n; f++) {
                if (i==f) A[i*n + f] = 1;
                else A[i*n + f] = 0;
            }
        }
        for (i=0; cust == m_aRatings2[r].CustId && r < MAX_RATINGS2; i++) {
            for (f=0; f<n; f++) Mi[f*ratingCount + i] = m_aMovieFeatures[f][m_aRatings2[r].MovieId];
            R[i] = double(m_aRatings2[r].Rating);
            r++;
        }

        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, ratingCount, 1.0, Mi, ratingCount, .05*ratingCount, A, n);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, ratingCount, 1.0, Mi, ratingCount, R, 1, 0.0, V, 1);
        info = dsysv(uplo, n, nrhs, A, n, ipiv, V, n, work, lwork);

        for (f=0; f<n; f++) m_aCustFeatures[f][cust] = V[f];

        delete[] Mi;
        delete[] R;
    }
}

//
// PredictRating
// - During training there is no need to loop through all of the features
// - Use a cache for the leading features and do a quick calculation for the trailing
// - The trailing can be optionally removed when calculating a new cache value
//
double Engine::PredictRating(short movieId, int custId, int feature, float cache, bool bTrailing)
{
    // Get cached value for old features or default to an average
    double sum = (cache > 0) ? cache : 1; //m_aMovies[movieId].PseudoAvg; 

    // Add contribution of current feature
    sum += m_aMovieFeatures[feature][movieId] * m_aCustFeatures[feature][custId];
    if (sum > 5) sum = 5;
    if (sum < 1) sum = 1;

    // Add up trailing defaults values
    if (bTrailing)
    {
        sum += (MAX_FEATURES-feature-1) * (INIT * INIT);
        if (sum > 5) sum = 5;
        if (sum < 1) sum = 1;
    }
    return sum;
}

//
// PredictRating
// - This version is used for calculating the final results
// - It loops through the entire list of finished features
//
double Engine::PredictRating(short movieId, int custId)
{
    double sum = 0; //m_aMovies[movieId].PseudoAvg;
//    double sum = 1; //m_aMovies[movieId].PseudoAvg;

    for (int f=0; f<MAX_FEATURES; f++) 
    {
        sum += m_aMovieFeatures[f][movieId] * m_aCustFeatures[f][custId];
        if (sum > 5) sum = 5;
        if (sum < 1) sum = 1;
    }

    return sum;
}

//-------------------------------------------------------------------
// Data Loading / Saving
//-------------------------------------------------------------------

//
// LoadHistory
// - Loop through all of the files in the training directory
//
void Engine::LoadHistory()
{
    char filename[15];
    // Loop through all of the files in the training directory
    for (int movieId = 1; movieId < MAX_MOVIES; movieId++) {
		sprintf(filename, "mv_%07d.txt", movieId);
        this->ProcessFile(filename);
    } 
}


//
// ProcessFile
// - Load a history file in the format:
//
//   <MovieId>:
//   <CustomerId>,<Rating>
//   <CustomerId>,<Rating>
//   ...
void Engine::ProcessFile(char* pwzFile)
{
    FILE *stream, *dataFile;
    char pwzBuffer[1000];
    sprintf(pwzBuffer,TRAINING_FILE,pwzFile);
    int custId, movieId, rating, pos = 0;
    printf("Processing file: %s\n", pwzBuffer);

    if (!(stream = fopen(pwzBuffer, "r"))) {
        cout << "Failed to open " << pwzBuffer << endl;
		getchar();
		return;
    }

    // First line is the movie id
    fgets(pwzBuffer, 1000, stream);
    ParseInt(pwzBuffer, (int)strlen(pwzBuffer), pos, movieId);
    m_aMovies[movieId].RatingCount = 0;
    m_aMovies[movieId].RatingSum = 0;

    // Get all remaining rows
	fgets(pwzBuffer, 1000, stream);
    while ( !feof( stream ) )
    {
        pos = 0;
        ParseInt(pwzBuffer, (int)strlen(pwzBuffer), pos, custId);
        ParseInt(pwzBuffer, (int)strlen(pwzBuffer), pos, rating);
        
        m_aRatings[m_nRatingCount].MovieId = (short)movieId;
        m_aRatings[m_nRatingCount].CustId = custId;
        m_aRatings[m_nRatingCount].Rating = (BYTE)rating;
        m_aRatings[m_nRatingCount].Cache = 0;
        m_nRatingCount++;

        fgets(pwzBuffer, 1000, stream);
    }

    // Cleanup
    fclose(stream);
}

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
void Engine::ProcessTest(const char* pwzFile)
{
    FILE *streamIn, *streamOut;
    char pwzBuffer[1000];
    int custId, movieId, pos = 0;
    double rating;
    bool bMovieRow;

    sprintf(pwzBuffer, TEST_PATH, pwzFile);
    printf("Processing test: %s\n", TEST_PATH);
    //printf("Processing test: %s\n", pwzBuffer);

    if (!(streamIn = fopen(pwzBuffer, "r"))) { 
        cout << "Failed to open " << pwzBuffer << endl;
		getchar();
		return;
    }
    if (!(streamOut = fopen(PREDICTION_FILE, "w"))) {
        cout << "Failed to open " << PREDICTION_FILE << endl;
		getchar();
		return;
    }

    fgets(pwzBuffer, 1000, streamIn);
    while ( !feof( streamIn ) )
    {
        bMovieRow = false;
        for (int i=0; i<(int)strlen(pwzBuffer); i++)
        {
            bMovieRow |= (pwzBuffer[i] == 58); 
        }

        pos = 0;
        if (bMovieRow)
        {
            ParseInt(pwzBuffer, (int)strlen(pwzBuffer), pos, movieId);

            // Write same row to results
            fputs(pwzBuffer,streamOut); 
        }
        else
        {
            ParseInt(pwzBuffer, (int)strlen(pwzBuffer), pos, custId);
            custId = m_mCustIds[custId];
            rating = PredictRating(movieId, custId);

            // Write predicted value
            sprintf(pwzBuffer,"%5.3f",rating);
            fputs(pwzBuffer,streamOut);
            fputs("\n",streamOut); 
        }

        //printf("Got Line: %d %d %d ", movieId, custId, rating);
        fgets(pwzBuffer, 1000, streamIn);
    }

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

bool Engine::ParseFloat(char* pwzBuffer, int nLength, int &nPosition, float& fValue)
{
    char pwzNumber[20];
    bool bResult = ReadNumber(pwzBuffer, nLength, nPosition, pwzNumber);
    fValue = (bResult) ? (float)atof(pwzNumber) : 0;
    return false;
}

bool Engine::ParseInt(char* pwzBuffer, int nLength, int &nPosition, int& nValue)
{
    char pwzNumber[20];
    bool bResult = ReadNumber(pwzBuffer, nLength, nPosition, pwzNumber);
    nValue = (bResult) ? atoi(pwzNumber) : 0;
    return bResult;
}

void Engine::LoadBinaries() {
    int i=0, elementSize, temp;
    FILE *ratingsFile, *moviesFile, *customersFile;
    size_t result;

    // movies array
cout << "loading movies" << endl;
	moviesFile = fopen("movies.bin","rb");
	elementSize = sizeof(int) + sizeof(int) + sizeof(double) + sizeof(double);
    for (i=0; i<MAX_MOVIES; i++)
    {
        Movie* movie = m_aMovies+i;
        result = fread(movie, elementSize, 1, moviesFile);
        if (result != 1) {
            cout << "failure reading movies.bin" << endl;
            getchar();
            return;
        }
		m_aMovieFeatures[0][i] = float(m_aMovies[i].RatingAvg);
		for (int f=1; f<MAX_FEATURES; f++) m_aMovieFeatures[f][i] = double(rand())/double(RAND_MAX)/5.0 - 0.1;
    }
	fclose(moviesFile);

    // customers array
cout << "loading customers" << endl;
    customersFile = fopen("customers.bin","rb");
    elementSize = sizeof(int) + sizeof(int) + sizeof(int);
    for (i=0; i<MAX_CUSTOMERS; i++) {
        Customer* customer = m_aCustomers+i;
        result = fread(customer, elementSize, 1, customersFile);
        if (result != 1) {
            cout << "failure reading customers.bin" << endl;
            getchar();
            return;
        }
        m_mCustIds[customer->CustomerId] = i;
		m_aCustFeatures[0][i] = float(m_aCustomers[i].RatingSum/(float(m_aCustomers[i].RatingCount)));
		for (int f=1; f<MAX_FEATURES; f++) m_aCustFeatures[f][i] = double(rand())/double(RAND_MAX)/5.0 - 0.1;
    }
    fclose(customersFile);

// load the ratings array
	cout << "loading ratings" << endl;
    ratingsFile = fopen("ratings.bin","rb");
    elementSize = sizeof(int) + sizeof(short) + sizeof(BYTE) + sizeof(float);
    for (i=0; i<MAX_RATINGS2; i++) {
		Data* rating = m_aRatings + i;
		result = fread(rating, elementSize, 1, ratingsFile);
        if (result != 1) {
            cout << "failure reading ratings.bin" << endl;
            getchar();
            return;
        }
		m_nRatingCount++;
    }
    fclose(ratingsFile);
// load ratings2 
    cout << "loading ratings2" << endl;
    if (MAX_RATINGS == MAX_RATINGS2) {
        ratingsFile = fopen("ratings2.bin","rb");
		elementSize = sizeof(int) + sizeof(short) + sizeof(BYTE) + sizeof(float);
		for (i=0; i<MAX_RATINGS2; i++) {
			Data* rating = m_aRatings2 + i;
			result = fread(rating, elementSize, 1, ratingsFile);
			if (result != 1) {
				cout << "failure reading ratings2.bin" << endl;
				getchar();
				return;
			}
		}
		fclose(ratingsFile);
    }
    else {
		ratingsFile = fopen("ratings.bin","rb");
		elementSize = sizeof(int) + sizeof(short) + sizeof(BYTE) + sizeof(float);
		for (i=0; i<m_nRatingCount; i++) {
			Data* rating = m_aRatings2 + i;
			result = fread(rating, elementSize, 1, ratingsFile);
			if (result != 1) {
				cout << "failure reading ratings.bin" << endl;
				getchar();
                return;
			}
            m_aMovies[rating->MovieId].RatingCount = 0;
            m_aCustomers[rating->CustId].RatingCount = 0;
		}
		fclose(ratingsFile);
		cout << "sorting..." << endl;
		sort(&m_aRatings2[0], &m_aRatings2[m_nRatingCount], CustomerLessThan);
    }

//redo the rating counts
    if (MAX_RATINGS != MAX_RATINGS2) {
        cout << "redoing rating counts" << endl;
        for (i=0; i<m_nRatingCount; i++) {
            Data* rating = m_aRatings2 + i;
            m_aMovies[rating->MovieId].RatingCount++;
            m_aCustomers[rating->CustId].RatingCount++;
        }
    }
} 

void Engine::CreateBinaries() {
    int i=0, elementSize;
    printf("Creating binary files\n");
    FILE *ratingsFile, *ratingsFile2, *moviesFile, *customersFile;

    // write out the ratings array
    ratingsFile = fopen("ratings.bin","w");
//    ratingsFile2 = fopen("ratings2.bin","w");
    elementSize = sizeof(int) + sizeof(short) + sizeof(BYTE) + sizeof(float);

    for (i=0; i<MAX_RATINGS; i++) {
        Data* rating = m_aRatings + i;
//    	Data* rating2 = m_aRatings2 + i;
        fwrite(rating, elementSize, 1, ratingsFile);
//    	fwrite(rating2, elementSize, 1, ratingsFile2);
        
/*        // Increment movie stats
        m_aMovies[rating->MovieId].RatingCount++;
        m_aMovies[rating->MovieId].RatingSum += rating->Rating;

        m_aCustomers[rating->CustId].RatingCount++;
        m_aCustomers[rating->CustId].RatingSum += rating->Rating;
*/
	}
    fclose(ratingsFile);
//    fclose(ratingsFile2);

    // movies array
	moviesFile = fopen("movies.bin","w");
	elementSize = sizeof(int) + sizeof(int) + sizeof(double) + sizeof(double);
    for (i=0; i<MAX_MOVIES; i++) {
        Movie* movie = m_aMovies+i;
        fwrite(movie, elementSize, 1, moviesFile);
    }
	fclose(moviesFile);

    // customers array
	customersFile = fopen("customers.bin","w");
	elementSize = sizeof(int) + sizeof(int) + sizeof(int);
    for (i=0; i<MAX_CUSTOMERS; i++) {
        Customer* customer = m_aCustomers+i;
        fwrite(customer, elementSize, 1, customersFile);
    }
	fclose(customersFile);
} 
