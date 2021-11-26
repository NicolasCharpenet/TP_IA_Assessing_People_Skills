using System;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using MBMLCommon;
using MBMLViews;


namespace StudentSkills
{
    class Program
    {
        private const string DataPath = @"Data/";
        /// <summary>
        /// The entry point for the application.
        /// The first argument, if present, sets the folder to save the output artifacts in.
        /// </summary>
        public static void Main(string[] args)
        {
            

            try
            {
                ToyWith3QuestionsAnd2Skills();

                LoopyExample();

                RealDataInference();


            }
            catch (Exception e)
            {
                Console.WriteLine($"\nAn unhandled exception was thrown:\n{e}");
            }
            finally
            {
                if (args.Length == 1)
                {
                                        
                    Console.WriteLine("Done");
                }
            }
        }

        public static void ToyWith3QuestionsAnd2Skills()
        {

            Experiment experiment = new Experiment
            {

                Inputs = LoadInputData("Toy3"),
                Model = new UnrolledModel
                {
                    Name = "ThreeQuestions",
                    ProbabilityOfGuess = 0.2,
                    ProbabilityOfNotMistake = 0.9,
                    ProbabilityOfSkillTrue = 0.5,
                    ShowFactorGraph = true

                }
            };

            experiment.Run();
            Console.WriteLine($" Iscorrect1: {experiment.Inputs.IsCorrect[0][0]},{experiment.Inputs.IsCorrect[1][0]},{experiment.Inputs.IsCorrect[2][0]},{experiment.Inputs.IsCorrect[3][0]},{experiment.Inputs.IsCorrect[4][0]},{experiment.Inputs.IsCorrect[5][0]},{experiment.Inputs.IsCorrect[6][0]},{experiment.Inputs.IsCorrect[7][0]}");
            Console.WriteLine($" Iscorrect2: {experiment.Inputs.IsCorrect[0][1]},{experiment.Inputs.IsCorrect[1][1]},{experiment.Inputs.IsCorrect[2][1]},{experiment.Inputs.IsCorrect[3][1]},{experiment.Inputs.IsCorrect[4][1]},{experiment.Inputs.IsCorrect[5][1]},{experiment.Inputs.IsCorrect[6][1]},{experiment.Inputs.IsCorrect[7][1]}");
            Console.WriteLine($" Iscorrect3: {experiment.Inputs.IsCorrect[0][2]},{experiment.Inputs.IsCorrect[1][2]},{experiment.Inputs.IsCorrect[2][2]},{experiment.Inputs.IsCorrect[3][2]},{experiment.Inputs.IsCorrect[4][2]},{experiment.Inputs.IsCorrect[5][2]},{experiment.Inputs.IsCorrect[6][2]},{experiment.Inputs.IsCorrect[7][2]}");
            Console.WriteLine($" P(csharp) : {experiment.Results.SkillsPosteriorMeans[0][0]},{experiment.Results.SkillsPosteriorMeans[1][0]},{experiment.Results.SkillsPosteriorMeans[2][0]},{experiment.Results.SkillsPosteriorMeans[3][0]},{experiment.Results.SkillsPosteriorMeans[4][0]},{experiment.Results.SkillsPosteriorMeans[5][0]},{experiment.Results.SkillsPosteriorMeans[6][0]},{experiment.Results.SkillsPosteriorMeans[7][0]}");
            Console.WriteLine($" P(sql)    : {experiment.Results.SkillsPosteriorMeans[0][1]},{experiment.Results.SkillsPosteriorMeans[1][1]},{experiment.Results.SkillsPosteriorMeans[2][1]},{experiment.Results.SkillsPosteriorMeans[3][1]},{experiment.Results.SkillsPosteriorMeans[4][1]},{experiment.Results.SkillsPosteriorMeans[5][1]},{experiment.Results.SkillsPosteriorMeans[6][1]},{experiment.Results.SkillsPosteriorMeans[7][1]}");
     }


        /// <summary>
        /// Results for loopy belief propagation section
        /// </summary>
        public static void LoopyExample()
        {
            Inputs inputs = LoadInputData("Toy4");
            Experiment loopyExperiment = new Experiment
            {
                // version without plates
                Inputs = inputs,
                Model = new UnrolledModel
                {
                    Name = "Loopy",
                    ProbabilityOfGuess = 0.2,
                    ProbabilityOfNotMistake = 0.9,
                    ProbabilityOfSkillTrue = 0.5,
                    ShowFactorGraph = true
                }
            };
            loopyExperiment.Run();
            Experiment exactLoopyExperiment = new Experiment
            {
                // exact results
                Inputs = inputs,
                Model = new UnrolledModel
                {
                    Name = "LoopyExact",
                    ProbabilityOfGuess = 0.2,
                    ProbabilityOfNotMistake = 0.9,
                    ProbabilityOfSkillTrue = 0.5,
                    ShowFactorGraph = true,
                    ExactInference = true
                }
            };
            exactLoopyExperiment.Run();
            
            //learningSkills.Run();

     }

        public static void RealDataInference()
        {
            // Define priors (fixed and learnt)
            const double ProbGuess = 0.2;
            const double ProbNotMistake = 0.9;
            const double ProbSkillTrue = 0.5;

            Inputs inputs = LoadInputData("InputData");

            
            // Original model (point mass priors)
            Experiment originalExperiment = new Experiment
            {
                Inputs = inputs,
                Model = new NoisyAndModel
                {
                    Name = "Original",
                    ProbabilityOfGuess = ProbGuess,
                    ProbabilityOfNotMistake = ProbNotMistake,
                    ProbabilityOfSkillTrue = ProbSkillTrue,
                    ShowFactorGraph = true,
                    IsReal = true
                }
            };
            originalExperiment.Run();

            // Sampled model using ground truth skills
            Experiment samleSkillsObservedExperiment = new Experiment
            {
                Inputs = originalExperiment.Model.SampleFromModel(
                            inputs, inputs.NumberOfPeople, new object[] { ProbSkillTrue, true }),
                Model = new NoisyAndModel
                {
                    Name = "SampleSkillsObserved",
                    ProbabilityOfGuess = ProbGuess,
                    ProbabilityOfNotMistake = ProbNotMistake,
                    ProbabilityOfSkillTrue = ProbSkillTrue,
                    IsReal = false
                }
            };

            samleSkillsObservedExperiment.Run();

            // Sampled model using sampled skills
            Experiment sampleSkillsSampledExperiment = new Experiment
            {
                Inputs = originalExperiment.Model.SampleFromModel(
                            inputs, inputs.NumberOfPeople, new object[] { ProbSkillTrue, false }),
                Model = new NoisyAndModel
                {
                    Name = "SampleSkillsSampled",
                    ProbabilityOfGuess = ProbGuess,
                    ProbabilityOfNotMistake = ProbNotMistake,
                    ProbabilityOfSkillTrue = ProbSkillTrue,
                    IsReal = false
                }
            };
            sampleSkillsSampledExperiment.Run();

            
            Beta guessPrior = BetaFromMeanAndTotalCount(0.25, 10);
            
            // Random model
            Experiment randomExperiment = new Experiment
            {
                Inputs = inputs,
                Model = new RandomModel
                {
                    Name = "Random",
                    ProbabilityOfGuess = 0.5,
                    ProbabilityOfNotMistake = 0.5,
                    ProbabilityOfSkillTrue = 0.5,
                    Index = 8,
                    ShowFactorGraph = true,
                    IsReal = false
                }
            };
            randomExperiment.Run();

            // Model with Beta prior over guess probabilities
            Experiment learnedExperiment = new Experiment
            {
                Inputs = inputs,
                Model = new LearnedNoisyAndModel
                {
                    Name = "Learned",
                    ProbabilityOfGuess = ProbGuess,
                    ProbabilityOfNotMistake = ProbNotMistake,
                    ProbabilityOfSkillTrue = ProbSkillTrue,
                    GuessPrior = guessPrior,
                    ShowFactorGraph = true,
                    IsReal = true
                }
            };
            learnedExperiment.Run();

            Experiment perfectExperiment = new Experiment
            {
                FullyObserved = true,
                Inputs = inputs,
                Model = new NoisyAndModel { Name = "Perfect", Index = 9, IsReal = false }
            };
            perfectExperiment.Run();

                  
        }


        #region BetaHelpers
        /// <summary>
        /// Betas from mean and total count.
        /// </summary>
        /// <param name="mean">The mean.</param>
        /// <param name="totalCount">The total count.</param>
        /// <returns>Beta distribution.</returns>
        private static Beta BetaFromMeanAndTotalCount(double mean, int totalCount)
        {
            return new Beta(mean * totalCount, (1 - mean) * totalCount);
        }
        #endregion






        #region File IO
        /// <summary>
        /// Loads the input data.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns>The input data.</returns>
        /// <exception cref="System.NullReferenceException">Failed to load input data
        /// or
        /// Failed to load quiz data</exception>
        public static Inputs LoadInputData(string filename)
        {
            var inputData = FileUtils.Load<Inputs>(DataPath, filename);

            if (inputData == null)
            {
                throw new NullReferenceException("Failed to load input data");
            }

            if (inputData.Quiz == null)
            {
                throw new NullReferenceException("Failed to load quiz data");
            }

            return inputData;
        }
        #endregion
    }

}


