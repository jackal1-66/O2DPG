#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <rapidjson/document.h>
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbApi.h"

// Static Ort::Env instance for multiple onnx model loading
static Ort::Env global_env(ORT_LOGGING_LEVEL_WARNING, "GlobalEnv");

// This class is responsible for loading the scaler parameters from a JSON file
// and applying the inverse transformation to the generated data.
struct Scaler
{
    std::vector<double> normal_min;
    std::vector<double> normal_max;
    std::vector<double> outlier_center;
    std::vector<double> outlier_scale;

    void load(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Error: Could not open scaler file!");
        }

        std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        rapidjson::Document doc;
        doc.Parse(json_str.c_str());

        if (doc.HasParseError())
        {
            throw std::runtime_error("Error: JSON parsing failed!");
        }

        normal_min = jsonArrayToVector(doc["normal"]["min"]);
        normal_max = jsonArrayToVector(doc["normal"]["max"]);
        outlier_center = jsonArrayToVector(doc["outlier"]["center"]);
        outlier_scale = jsonArrayToVector(doc["outlier"]["scale"]);
    }

    std::vector<double> inverse_transform(const std::vector<double> &input)
    {
        std::vector<double> output;
        for (int i = 0; i < input.size(); ++i)
        {
            if (i < input.size() - 2)
                output.push_back(input[i] * (normal_max[i] - normal_min[i]) + normal_min[i]);
            else
                output.push_back(input[i] * outlier_scale[i - (input.size() - 2)] + outlier_center[i - (input.size() - 2)]);
        }

        return output;
    }

private:
    std::vector<double> jsonArrayToVector(const rapidjson::Value &jsonArray)
    {
        std::vector<double> vec;
        for (int i = 0; i < jsonArray.Size(); ++i)
        {
            vec.push_back(jsonArray[i].GetDouble());
        }
        return vec;
    }
};

// This class loads the ONNX model and generates samples using it.
class ONNXGenerator
{
public:
    ONNXGenerator(Ort::Env &shared_env, const std::string &model_path)
        : env(shared_env), session(env, model_path.c_str(), Ort::SessionOptions{})
    {
        // Create session options
        Ort::SessionOptions session_options;
        session = Ort::Session(env, model_path.c_str(), session_options);
    }

    std::vector<double> generate_sample()
    {
        Ort::AllocatorWithDefaultOptions allocator;

        // Generate a latent vector (z)
        std::vector<float> z(100);
        for (auto &v : z)
            v = rand_gen.Gaus(0.0, 1.0);

        // Prepare input tensor
        std::vector<int64_t> input_shape = {1, 100};
        // Get memory information
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensor correctly
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, z.data(), z.size(), input_shape.data(), input_shape.size());
        // Run inference
        const char *input_names[] = {"z"};
        const char *output_names[] = {"output"};
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // Extract output
        float *output_data = output_tensors.front().GetTensorMutableData<float>();
        // Get the size of the output tensor
        auto output_tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
        size_t output_data_size = output_tensor_info.GetElementCount(); // Total number of elements in the tensor
        std::vector<double> output;
        for (int i = 0; i < output_data_size; ++i)
        {
            output.push_back(output_data[i]);
        }

        return output;
    }

private:
    Ort::Env &env;
    Ort::Session session;
    TRandom3 rand_gen;
};

namespace o2
{
namespace eventgen
{

class GenTPCLoopers : public Generator
{
    public:
        GenTPCLoopers(std::string model_pairs = "tpcloopmodel.onnx", std::string model_compton = "tpcloopmodelcompton.onnx",
                      std::string poisson = "poisson.csv", std::string gauss = "gauss.csv", std::string scaler_pair = "scaler_pair.json",
                      std::string scaler_compton = "scaler_compton.json")
        {
            // Checking if the model files exist and are not empty
            std::ifstream model_file[2];
            model_file[0].open(model_pairs);
            model_file[1].open(model_compton);
            if (!model_file[0].is_open() || model_file[0].peek() == std::ifstream::traits_type::eof())
            {
                LOG(fatal) << "Error: Pairs model file is empty or does not exist!";
                exit(1);
            }
            if (!model_file[1].is_open() || model_file[1].peek() == std::ifstream::traits_type::eof())
            {
                LOG(fatal) << "Error: Compton model file is empty or does not exist!";
                exit(1);
            }
            model_file[0].close();
            model_file[1].close();
            // Checking if the scaler files exist and are not empty
            std::ifstream scaler_file[2];
            scaler_file[0].open(scaler_pair);
            scaler_file[1].open(scaler_compton);
            if (!scaler_file[0].is_open() || scaler_file[0].peek() == std::ifstream::traits_type::eof())
            {
                LOG(fatal) << "Error: Pairs scaler file is empty or does not exist!";
                exit(1);
            }
            if (!scaler_file[1].is_open() || scaler_file[1].peek() == std::ifstream::traits_type::eof())
            {
                LOG(fatal) << "Error: Compton scaler file is empty or does not exist!";
                exit(1);
            }
            scaler_file[0].close();
            scaler_file[1].close();
            // Checking if the poisson file exists and it's not empty
            if (poisson != "")
            {
                std::ifstream poisson_file(poisson);
                if (!poisson_file.is_open() || poisson_file.peek() == std::ifstream::traits_type::eof())
                {
                    LOG(fatal) << "Error: Poisson file is empty or does not exist!";
                    exit(1);
                }
                else
                {
                    poisson_file >> mPoisson[0] >> mPoisson[1] >> mPoisson[2];
                    poisson_file.close();
                    mPoissonSet = true;
                }
            }
            // Checking if the gauss file exists and it's not empty
            if (gauss != "")
            {
                std::ifstream gauss_file(gauss);
                if (!gauss_file.is_open() || gauss_file.peek() == std::ifstream::traits_type::eof())
                {
                    LOG(fatal) << "Error: Gauss file is empty or does not exist!";
                    exit(1);
                }
                else
                {
                    gauss_file >> mGauss[0] >> mGauss[1] >> mGauss[2] >> mGauss[3];
                    gauss_file.close();
                    mGaussSet = true;
                }
            }
            mONNX_pair = std::make_unique<ONNXGenerator>(global_env, model_pairs);
            mScaler_pair = std::make_unique<Scaler>();
            mScaler_pair->load(scaler_pair);
            mONNX_compton = std::make_unique<ONNXGenerator>(global_env, model_compton);
            mScaler_compton = std::make_unique<Scaler>();
            mScaler_compton->load(scaler_compton);
            Generator::setTimeUnit(1.0);
            Generator::setPositionUnit(1.0);
        }
    
        Bool_t generateEvent() override
        {   
            // Clear the vector of pairs
            mGenPairs.clear();
            // Clear the vector of compton electrons
            mGenElectrons.clear();
            // Set number of loopers if poissonian params are available
            if (mPoissonSet)
            {
                mNLoopersPairs = static_cast<short int>(std::round(mMultiplier[0] * PoissonPairs()));
            }
            if (mGaussSet)
            {
                mNLoopersCompton = static_cast<short int>(std::round(mMultiplier[1] * GaussianElectrons()));
            }
            // Generate pairs
            for (int i = 0; i < mNLoopersPairs; ++i)
            {
                std::vector<double> pair = mONNX_pair->generate_sample();
                // Apply the inverse transformation using the scaler
                std::vector<double> transformed_pair = mScaler_pair->inverse_transform(pair);
                mGenPairs.push_back(transformed_pair);
            }
            // Generate compton electrons
            for (int i = 0; i < mNLoopersCompton; ++i)
            {
                std::vector<double> electron = mONNX_compton->generate_sample();
                // Apply the inverse transformation using the scaler
                std::vector<double> transformed_electron = mScaler_compton->inverse_transform(electron);
                mGenElectrons.push_back(transformed_electron);
            }
            return true;
        }

        Bool_t generateEvent(TH1D* t_hist)
        {
            // Clear the vector of pairs
            mGenPairs.clear();
            // Clear the vector of compton electrons
            mGenElectrons.clear();
            // Set number of loopers if poissonian params are available
            if (mPoissonSet)
            {
                mNLoopersPairs = static_cast<short int>(std::round(mMultiplier[0] * PoissonPairs()));
            }
            if (mGaussSet)
            {
                mNLoopersCompton = static_cast<short int>(std::round(mMultiplier[1] * GaussianElectrons()));
            }
            // Find the time histogram edge
            double time_constraint = t_hist->GetXaxis()->GetXmax();
            LOG(info) << "Time constraint for loopers: " << time_constraint << " ns";
            // Generate pairs
            for (int i = 0; i < mNLoopersPairs; ++i)
            {
                std::vector<double> pair = mONNX_pair->generate_sample();
                // Apply the inverse transformation using the scaler
                std::vector<double> transformed_pair = mScaler_pair->inverse_transform(pair);
                // Check if the time constraint is satisfied
                while ((transformed_pair[9] / 1e9) > time_constraint)
                {
                    transformed_pair[9] = t_hist->GetRandom() * 1e9; // Regenerate time if it exceeds the constraint, multiplied by 1e9 to account for ratio in importParticles
                }
                LOG(info) << "Transformed pair time: " << transformed_pair[9] / 1e9 << " ns";
                mGenPairs.push_back(transformed_pair);
            }
            // Generate compton electrons
            for (int i = 0; i < mNLoopersCompton; ++i)
            {
                std::vector<double> electron = mONNX_compton->generate_sample();
                // Apply the inverse transformation using the scaler
                std::vector<double> transformed_electron = mScaler_compton->inverse_transform(electron);
                while ((transformed_electron[6] / 1e9) > time_constraint)
                {
                    transformed_electron[6] = t_hist->GetRandom() * 1e9; // Regenerate time if it exceeds the constraint, multiplied by 1e9 to account for ratio in importParticles
                }
                LOG(info) << "Transformed electron time: " << transformed_electron[6] / 1e9 << " ns";
                mGenElectrons.push_back(transformed_electron);
            }
            return true;
        }

        Bool_t importParticles() override
        {
            // Get looper pairs from the event
            for (auto &pair : mGenPairs)
            {
                double px_e, py_e, pz_e, px_p, py_p, pz_p;
                double vx, vy, vz, time;
                double e_etot, p_etot;
                px_e = pair[0];
                py_e = pair[1];
                pz_e = pair[2];
                px_p = pair[3];
                py_p = pair[4];
                pz_p = pair[5];
                vx = pair[6];
                vy = pair[7];
                vz = pair[8];
                time = pair[9];
                e_etot = TMath::Sqrt(px_e * px_e + py_e * py_e + pz_e * pz_e + mMass_e * mMass_e);
                p_etot = TMath::Sqrt(px_p * px_p + py_p * py_p + pz_p * pz_p + mMass_p * mMass_p);
                // Push the electron
                TParticle electron(11, 1, -1, -1, -1, -1, px_e, py_e, pz_e, e_etot, vx, vy, vz, time / 1e9);
                electron.SetStatusCode(o2::mcgenstatus::MCGenStatusEncoding(electron.GetStatusCode(), 0).fullEncoding);
                electron.SetBit(ParticleStatus::kToBeDone, //
                                o2::mcgenstatus::getHepMCStatusCode(electron.GetStatusCode()) == 1);
                mParticles.push_back(electron);
                // Push the positron
                TParticle positron(-11, 1, -1, -1, -1, -1, px_p, py_p, pz_p, p_etot, vx, vy, vz, time / 1e9);
                positron.SetStatusCode(o2::mcgenstatus::MCGenStatusEncoding(positron.GetStatusCode(), 0).fullEncoding);
                positron.SetBit(ParticleStatus::kToBeDone, //
                                o2::mcgenstatus::getHepMCStatusCode(positron.GetStatusCode()) == 1);
                mParticles.push_back(positron);
            }
            // Get compton electrons from the event
            for (auto &compton : mGenElectrons)
            {
                double px, py, pz;
                double vx, vy, vz, time;
                double etot;
                px = compton[0];
                py = compton[1];
                pz = compton[2];
                vx = compton[3];
                vy = compton[4];
                vz = compton[5];
                time = compton[6];
                etot = TMath::Sqrt(px * px + py * py + pz * pz + mMass_e * mMass_e);
                // Push the electron
                TParticle electron(11, 1, -1, -1, -1, -1, px, py, pz, etot, vx, vy, vz, time / 1e9);
                electron.SetStatusCode(o2::mcgenstatus::MCGenStatusEncoding(electron.GetStatusCode(), 0).fullEncoding);
                electron.SetBit(ParticleStatus::kToBeDone, //
                                o2::mcgenstatus::getHepMCStatusCode(electron.GetStatusCode()) == 1);
                mParticles.push_back(electron);
            }

            return true;
        }

        short int PoissonPairs()
        {
            short int poissonValue;
            do
            {
                // Generate a Poisson-distributed random number with mean mPoisson[0]
                poissonValue = mRandGen.Poisson(mPoisson[0]);
            } while (poissonValue < mPoisson[1] || poissonValue > mPoisson[2]); // Regenerate if out of range

            return poissonValue;
        }

        short int GaussianElectrons()
        {
            short int gaussValue;
            do
            {
                // Generate a Normal-distributed random number with mean mGass[0] and stddev mGauss[1]
                gaussValue = mRandGen.Gaus(mGauss[0], mGauss[1]);
            } while (gaussValue < mGauss[2] || gaussValue > mGauss[3]); // Regenerate if out of range

            return gaussValue;
        }

        void SetNLoopers(short int &nsig_pair, short int &nsig_compton)
        {
            if(mPoissonSet) {
                LOG(info) << "Poissonian parameters correctly loaded.";
            } else {
                mNLoopersPairs = nsig_pair;
            }
            if(mGaussSet) {
                LOG(info) << "Gaussian parameters correctly loaded.";
            } else {
                mNLoopersCompton = nsig_compton;
            }
        }

        void SetMultiplier(std::array<float, 2> &mult)
        {
            // Multipliers will work only if the poissonian and gaussian parameters are set
            // otherwise they will be ignored
            if (mult[0] < 0 || mult[1] < 0)
            {
                LOG(fatal) << "Error: Multiplier values must be non-negative!";
                exit(1);
            } else {
                LOG(info) << "Multiplier values set to: Pair = " << mult[0] << ", Compton = " << mult[1];
                mMultiplier[0] = mult[0];
                mMultiplier[1] = mult[1];
            }
        }

    private:
        std::unique_ptr<ONNXGenerator> mONNX_pair = nullptr;
        std::unique_ptr<ONNXGenerator> mONNX_compton = nullptr;
        std::unique_ptr<Scaler> mScaler_pair = nullptr;
        std::unique_ptr<Scaler> mScaler_compton = nullptr;
        double mPoisson[3] = {0.0, 0.0, 0.0}; // Mu, Min and Max of Poissonian
        double mGauss[4] = {0.0, 0.0, 0.0, 0.0}; // Mean, Std, Min, Max
        std::vector<std::vector<double>> mGenPairs;
        std::vector<std::vector<double>> mGenElectrons;
        short int mNLoopersPairs = -1;
        short int mNLoopersCompton = -1;
        std::array<float, 2> mMultiplier = {1., 1.};
        bool mPoissonSet = false;
        bool mGaussSet = false;
        // Random number generator
        TRandom3 mRandGen;
        // Masses of the electrons and positrons
        TDatabasePDG *mPDG = TDatabasePDG::Instance();
        double mMass_e = mPDG->GetParticle(11)->Mass();
        double mMass_p = mPDG->GetParticle(-11)->Mass();
};

class GenLoopersInjector : public Generator
{
    public:
        GenLoopersInjector(std::string kineFN = "genevents_Kine.root", std::string model_pairs = "tpcloopmodel.onnx", std::string model_compton = "tpcloopmodelcompton.onnx",
                           std::string scaler_pair = "scaler_pair.json", std::string scaler_compton = "scaler_compton.json", std::string poisson = "", std::string gauss = "")
        {
            mKineGen = std::make_unique<GeneratorFromO2Kine>(kineFN.c_str());
            mGenTPCLoopers = std::make_unique<GenTPCLoopers>(model_pairs, model_compton, poisson, gauss, scaler_pair, scaler_compton);
            Generator::setTimeUnit(1.0);
            Generator::setPositionUnit(1.0);
            Generator::setMomentumUnit(1.0);
            Generator::setEnergyUnit(1.0);
        }

        void setAdaptiveLoopers(Bool_t &adaptive)
        {
            mAdaptiveLoopers = adaptive;
            LOG(info) << "Adaptive loopers: " << (mAdaptiveLoopers ? "ON" : "OFF");
        }

        void setTimeConstraint(Bool_t &constraint)
        {
            mTimeConstraint = constraint;
            LOG(info) << "Time constraint: " << (mTimeConstraint ? "ON" : "OFF");
        }

        void setLoopsFractions(float &fraction, float &fractionPairs)
        {
            if (fraction < 0 || fraction >= 1)
            {
                LOG(fatal) << "Error: Loops fraction must be in the range [0, 1).";
                exit(1);
            }
            mLoopsFraction = fraction;
            if (fractionPairs < 0 || fractionPairs > 1)
            {
                LOG(fatal) << "Error: Loops fraction for pairs must be in the range [0, 1].";
                exit(1);
            }
            mLoopsFractionPairs = fractionPairs;
            LOG(info) << "Pairs fraction set to: " << mLoopsFraction;
        }

        Bool_t generateEvent() override
        {
            // Trivial, real work in importParticles
            return true;
        }

        std::vector<TParticle> genProcessor(Generator* gen)
        {
            auto unit_transformer = [](auto &p, auto pos_unit, auto time_unit, auto en_unit, auto mom_unit)
            {
                p.SetMomentum(p.Px() * mom_unit, p.Py() * mom_unit, p.Pz() * mom_unit, p.Energy() * en_unit);
                p.SetProductionVertex(p.Vx() * pos_unit, p.Vy() * pos_unit, p.Vz() * pos_unit, p.T() * time_unit);
            };

            auto index_transformer = [](auto &p, int offset)
            {
                for (int i = 0; i < 2; ++i)
                {
                    if (p.GetMother(i) != -1)
                    {
                        const auto newindex = p.GetMother(i) + offset;
                        p.SetMother(i, newindex);
                    }
                }
                if (p.GetNDaughters() > 0)
                {
                    for (int i = 0; i < 2; ++i)
                    {
                        const auto newindex = p.GetDaughter(i) + offset;
                        p.SetDaughter(i, newindex);
                    }
                }
            };
            auto parts = gen->getParticles();
            auto time_unit = gen->getTimeUnit();
            auto pos_unit = gen->getPositionUnit();
            auto mom_unit = gen->getMomentumUnit();
            auto energy_unit = gen->getEnergyUnit();
            for (auto &p : parts)
            {
                // apply the mother-daugher index transformation
                index_transformer(p, mParticles.size());
                // apply unit transformation of sub-generator
                unit_transformer(p, pos_unit, time_unit, energy_unit, mom_unit);
            }
            // Print Maximum values of time
            double maxTime = 0.0;
            double minTime = 0.0;
            for (const auto &p : parts) {
                if (p.T() > maxTime) {
                    maxTime = p.T();
                }
                if (p.T() < minTime) {
                    minTime = p.T();
                }
            }
            LOG(info) << "Maximum time in the event: " << maxTime;
            LOG(info) << "Minimum time in the event: " << minTime;
            return parts;
        }

        Bool_t importParticles() override
        {
            mParticles.clear(); // Clear the particles stack before importing new ones
            // Combination of import particles from GeneratorFromO2Kine and GenTPCLoopers
            mKineGen->clearParticles(); // Clear particles from O2 Kinematics generator
            mGenTPCLoopers->clearParticles(); // Clear particles from GenTPCLoopers
            auto stat1 = mKineGen->importParticles();
            if (stat1)
            {
                // Include particles from O2 Kinematics generator
                auto kineparts = genProcessor(mKineGen.get());
                LOG(info) << "Size of kineparts: " << kineparts.size();
                mParticles.insert(mParticles.end(), kineparts.begin(), kineparts.end());
                LOG(info) << "Size mParticles at kine stage " << mParticles.size();
                double maxTime = 0.0;
                // Find the maximum time in the imported particles
                for (const auto &p : mParticles) {
                    if (p.T() > maxTime) {
                        maxTime = p.T();
                    }
                }
                // Create a histogram containing original generator time values
                auto time_hist = std::make_unique<TH1D>("time_hist", "Time Histogram", 1000, 0., maxTime);
                for (const auto &p : mParticles) {
                    time_hist->Fill(p.T());
                }
                // Check size of mParticles stack and set loopers accordingly if mAdaptiveLoopers is true
                if (mAdaptiveLoopers) {
                    int nParticles = mParticles.size();
                    if (nParticles > 0)
                    {
                        // Calculate the number of loopers to inject adaptively
                        short int nLoopers = static_cast<short int>(std::round((nParticles * mLoopsFraction) / (1 - mLoopsFractionPairs)));
                        short int nLoopersPairs = static_cast<short int>(std::round(nLoopers * mLoopsFractionPairs));
                        short int nLoopersCompton = nLoopers - nLoopersPairs;
                        mGenTPCLoopers->SetNLoopers(nLoopersPairs, nLoopersCompton);
                        LOG(info) << "Adaptive loopers: " << nLoopers << " (pairs: " << nLoopersPairs << ", compton: " << nLoopersCompton << ")";
                    } else {
                        LOG(info) << "No particles found in O2 Kinematics, no loopers will be generated";
                        return false;
                    }
                }
                // Generate loopers using GenTPCLoopers
                // this is valid also when number of loopers is fixed
                if (mTimeConstraint) {
                    mGenTPCLoopers->generateEvent(time_hist.get());
                } else {
                    mGenTPCLoopers->generateEvent();
                }
                auto stat2 = mGenTPCLoopers->importParticles();
                if (!stat2) {
                    LOG(error) << "Failed to import particles from GenTPCLoopers";
                    return false;
                }
                // Include particles from GenTPCLoopers
                auto loopers = genProcessor(mGenTPCLoopers.get());
                LOG(info) << "Size of loopers: " << loopers.size();
                mParticles.insert(mParticles.end(), loopers.begin(), loopers.end());
                LOG(info) << "Size mParticles at loopers stage " << mParticles.size();
            } else {
                LOG(error) << "Failed to import particles from O2 Kinematics";
                return false;
            }
            return true;
        }

    private:
        std::unique_ptr<GeneratorFromO2Kine> mKineGen = nullptr; // Instance of GeneratorFromO2Kine to read particles from O2 kinematics file
        std::unique_ptr<GenTPCLoopers> mGenTPCLoopers = nullptr; // Instance of GenTPCLoopers to generate loopers
        Bool_t mAdaptiveLoopers = true; // Flag to indicate if adaptive loopers are used
        float mLoopsFraction = 0.05; // Fraction of loopers to be injected adaptively
        float mLoopsFractionPairs = 0.08; // Fraction of loopers from Pairs
        Bool_t mTimeConstraint = true; // Flag to indicate if time constraint is applied
        double mTimeConstraintValue = 0.0; // Time constraint value, adaptively set based on the maximum time of the main generator events
};

} // namespace eventgen
} // namespace o2

// ONNX model files can be local, on AliEn or in the ALICE CCDB.
// For local and alien files it is mandatory to provide the filenames, for the CCDB instead the
// path to the object in the CCDB is sufficient. The model files will be downloaded locally.
// Example of CCDB path: "ccdb://Users/n/name/test"
// Example of alien path: "alien:///alice/cern.ch/user/n/name/test/test.onnx"
FairGenerator *
    Generator_TPCLoopers(std::string model_pairs = "tpcloopmodel.onnx", std::string model_compton = "tpcloopmodelcompton.onnx",
                         std::string poisson = "poisson.csv", std::string gauss = "gauss.csv", std::string scaler_pair = "scaler_pair.json",
                         std::string scaler_compton = "scaler_compton.json", std::array<float, 2> mult = {1., 1.}, short int nloopers_pairs = 1,
                         short int nloopers_compton = 1)
{
    // Expand all environment paths
    model_pairs = gSystem->ExpandPathName(model_pairs.c_str());
    model_compton = gSystem->ExpandPathName(model_compton.c_str());
    poisson = gSystem->ExpandPathName(poisson.c_str());
    gauss = gSystem->ExpandPathName(gauss.c_str());
    scaler_pair = gSystem->ExpandPathName(scaler_pair.c_str());
    scaler_compton = gSystem->ExpandPathName(scaler_compton.c_str());
    const std::array<std::string, 2> models = {model_pairs, model_compton};
    const std::array<std::string, 2> local_names = {"WGANpair.onnx", "WGANcompton.onnx"};
    const std::array<bool, 2> isAlien = {models[0].starts_with("alien://"), models[1].starts_with("alien://")};
    const std::array<bool, 2> isCCDB = {models[0].starts_with("ccdb://"), models[1].starts_with("ccdb://")};
    if (std::any_of(isAlien.begin(), isAlien.end(), [](bool v) { return v; }))
    {
        if (!gGrid) {
            TGrid::Connect("alien://");
            if (!gGrid) {
                LOG(fatal) << "AliEn connection failed, check token.";
                exit(1);
            }
        }
        for (size_t i = 0; i < models.size(); ++i)
        {
            if (isAlien[i] && !TFile::Cp(models[i].c_str(), local_names[i].c_str()))
            {
                LOG(fatal) << "Error: Model file " << models[i] << " does not exist!";
                exit(1);
            }
        }
    }
    if (std::any_of(isCCDB.begin(), isCCDB.end(), [](bool v) { return v; }))
    {
        o2::ccdb::CcdbApi ccdb_api;
        ccdb_api.init("http://alice-ccdb.cern.ch");
        for (size_t i = 0; i < models.size(); ++i)
        {
            if (isCCDB[i])
            {
                auto model_path = models[i].substr(7); // Remove "ccdb://"
                // Treat filename if provided in the CCDB path
                auto extension = model_path.find(".onnx");
                if (extension != std::string::npos)
                {
                    auto last_slash = model_path.find_last_of('/');
                    model_path = model_path.substr(0, last_slash);
                }
                std::map<std::string, std::string> filter;
                if(!ccdb_api.retrieveBlob(model_path, "./" , filter, o2::ccdb::getCurrentTimestamp(), false, local_names[i].c_str()))
                {
                    LOG(fatal) << "Error: issues in retrieving " << model_path << " from CCDB!";
                    exit(1);
                }
            }
        }
    }
    model_pairs = isAlien[0] || isCCDB[0] ? local_names[0] : model_pairs;
    model_compton = isAlien[1] || isCCDB[1] ? local_names[1] : model_compton;
    auto generator = new o2::eventgen::GenTPCLoopers(model_pairs, model_compton, poisson, gauss, scaler_pair, scaler_compton);
    generator->SetNLoopers(nloopers_pairs, nloopers_compton);
    generator->SetMultiplier(mult);
    return generator;
}

// Loopers injector to O2 kinematics file
// Loopers are considered adaptive by default, meaning that the number of loopers is determined by the number of particles in the kinematics file per event
FairGenerator *
GeneratorLoopersInjector(std::string kineFileName = "genevents_Kine.root", std::string model_pairs = "tpcloopmodel.onnx", std::string model_compton = "tpcloopmodelcompton.onnx",
                     std::string scaler_pair = "scaler_pair.json", std::string scaler_compton = "scaler_compton.json", bool time_constraint = true, float loopers_fraction = 0.05, float fraction_pairs = 0.08)
{
    // Expand all environment paths
    model_pairs = gSystem->ExpandPathName(model_pairs.c_str());
    model_compton = gSystem->ExpandPathName(model_compton.c_str());
    scaler_pair = gSystem->ExpandPathName(scaler_pair.c_str());
    scaler_compton = gSystem->ExpandPathName(scaler_compton.c_str());
    const std::array<std::string, 2> models = {model_pairs, model_compton};
    const std::array<std::string, 2> local_names = {"WGANpair.onnx", "WGANcompton.onnx"};
    const std::array<bool, 2> isAlien = {models[0].starts_with("alien://"), models[1].starts_with("alien://")};
    const std::array<bool, 2> isCCDB = {models[0].starts_with("ccdb://"), models[1].starts_with("ccdb://")};
    if (std::any_of(isAlien.begin(), isAlien.end(), [](bool v)
                    { return v; }))
    {
        if (!gGrid)
        {
            TGrid::Connect("alien://");
            if (!gGrid)
            {
                LOG(fatal) << "AliEn connection failed, check token.";
                exit(1);
            }
        }
        for (size_t i = 0; i < models.size(); ++i)
        {
            if (isAlien[i] && !TFile::Cp(models[i].c_str(), local_names[i].c_str()))
            {
                LOG(fatal) << "Error: Model file " << models[i] << " does not exist!";
                exit(1);
            }
        }
    }
    if (std::any_of(isCCDB.begin(), isCCDB.end(), [](bool v)
                    { return v; }))
    {
        o2::ccdb::CcdbApi ccdb_api;
        ccdb_api.init("http://alice-ccdb.cern.ch");
        for (size_t i = 0; i < models.size(); ++i)
        {
            if (isCCDB[i])
            {
                auto model_path = models[i].substr(7); // Remove "ccdb://"
                // Treat filename if provided in the CCDB path
                auto extension = model_path.find(".onnx");
                if (extension != std::string::npos)
                {
                    auto last_slash = model_path.find_last_of('/');
                    model_path = model_path.substr(0, last_slash);
                }
                std::map<std::string, std::string> filter;
                if (!ccdb_api.retrieveBlob(model_path, "./", filter, o2::ccdb::getCurrentTimestamp(), false, local_names[i].c_str()))
                {
                    LOG(fatal) << "Error: issues in retrieving " << model_path << " from CCDB!";
                    exit(1);
                }
            }
        }
    }
    model_pairs = isAlien[0] || isCCDB[0] ? local_names[0] : model_pairs;
    model_compton = isAlien[1] || isCCDB[1] ? local_names[1] : model_compton;
    auto generator = new o2::eventgen::GenLoopersInjector(kineFileName, model_pairs, model_compton, scaler_pair, scaler_compton);
    auto& extParams = o2::eventgen::GeneratorExternalParam::Instance();
    auto config = extParams.config;
    // Parse external configuration for loopers fractions if provided
    if (!config.empty()) {
        LOG(info) << "Using external configuration for loopers fractions: " << extParams.config;
        // Try to parse as "<loopers_fraction>-<fraction_pairs>" or single value
        size_t dash = config.find('-');
        try {
            if (dash == std::string::npos) {
                float frac = std::stof(config);
                if (frac >= 0 && frac < 1) {
                    generator->setLoopsFractions(frac, fraction_pairs);
                } else {
                    throw std::invalid_argument("Out of range");
                }
            } else {
                float frac1 = std::stof(config.substr(0, dash));
                float frac2 = std::stof(config.substr(dash + 1));
                if (frac1 >= 0 && frac1 < 1 && frac2 >= 0 && frac2 <= 1) {
                    generator->setLoopsFractions(frac1, frac2);
                } else {
                    throw std::invalid_argument("Out of range");
                }
            }
        } catch (...) {
            LOG(warn) << "Invalid external configuration for loopers fractions: " << extParams.config;
            LOG(warn) << "Expected format: <loopers_fraction>-<fraction_pairs> or a single value between [0,1).";
            LOG(warn) << "Using default values instead.";
            generator->setLoopsFractions(loopers_fraction, fraction_pairs);
        }
    } else {
        generator->setLoopsFractions(loopers_fraction, fraction_pairs);
    }
    // Set adaptive time constraint flag
    generator->setTimeConstraint(time_constraint);
    return generator;
}