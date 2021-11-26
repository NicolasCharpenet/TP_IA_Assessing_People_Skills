using System;

namespace StudentSkills
{
    /// <summary>
    /// Container class for all of the data associated with the quiz (quiz data, input data, results data, metrics data)
    /// </summary>
    [Serializable]
    public class Experiment
    {
        /// <summary>
        /// Gets the model name.
        /// </summary>
        public string ModelName
        {
            get
            {
                return this.Model != null ? this.Model.Name : "Undefined";
            }
        }

        /// <summary>
        /// Gets or sets the inputs.
        /// </summary>
        public Inputs Inputs { get; set; }

        /// <summary>
        /// Gets or sets the results.
        /// </summary>
        public Results Results { get; set; }

        /// <summary>
        /// Gets or sets the model.
        /// </summary>
        public NoisyAndModel Model { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether fully observed.
        /// </summary>
        public bool FullyObserved { get; set; }
        
        /// <summary>
        /// Run the experiment
        /// </summary>
        /// <exception cref="System.NullReferenceException">Inputs and Model must not be null</exception>
        /// <exception cref="NullReferenceException"></exception>
        public void Run()
        {
            if (this.Inputs == null || this.Model == null)
            {
                throw new NullReferenceException("Inputs and Model must not be null");
            }

            this.Model.ConstructModel();

            Results results = new Results();

            if (this.FullyObserved)
            {
                this.Model.SetObservedValues(this.Inputs, true, true);
                this.Model.DoInference(ref results);
            }
            else
            {
                // Run inference on has skills
                try
                {
                    this.Model.SetObservedValues(this.Inputs, false, true);
                    this.Model.DoInference(ref results);
                }
                catch (NotSupportedException)
                {
                }
            }

            this.Results = results;

            
        }
    }
}
