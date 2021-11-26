namespace StudentSkills
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MBMLViews;

    using Microsoft.ML.Probabilistic.Distributions;

    /// <summary>
    /// Class to hold inference results
    /// </summary>
    public class Results
    {
        /// <summary>
        /// Gets or sets the skills posteriors.
        /// </summary>
        /// <value>
        /// The skills posteriors.
        /// </value>
        public Bernoulli[][] SkillsPosteriors { get; set; }

        /// <summary>
        /// Gets or sets the guess posteriors.
        /// </summary>
        /// <value>
        /// The guess posteriors.
        /// </value>
        public IList<Beta> GuessPosteriors { get; set; }
        
        /// <summary>
        /// Gets the guess posteriors as dictionary.
        /// </summary>
        /// <value>
        /// The guess posteriors as dictionary.
        /// </value>
        public Dictionary<string, object> GuessPosteriorsAsDictionary
        {
            get
            {
                if (this.GuessPosteriors == null)
                {
                    return null;
                }

                return new Dictionary<string, object>
                    {
                        {
                            "Question",
                            Enumerable.Range(1, this.GuessPosteriors.Count)
                                        .Select(ia => string.Format("{0}", ia))
                                        .ToArray()
                        },
                        { "Posterior", this.GuessPosteriors }
                    };
            }
        }
        
        /// <summary>
        /// Gets or sets the is correct posteriors.
        /// </summary>
        /// <value>
        /// The is correct posteriors.
        /// </value>
        public Bernoulli[][] IsCorrectPosteriors { get; set; }

        /// <summary>
        /// Gets the skills posterior means.
        /// </summary>
        /// <value>
        /// The skills posterior means.
        /// </value>
        public double[][] SkillsPosteriorMeans
        {
            get
            {
                return this.SkillsPosteriors == null ? null : this.SkillsPosteriors.GetMeans();
            }
        }

        /// <summary>
        /// Gets the guess posterior means.
        /// </summary>
        /// <value>
        /// The guess posterior means.
        /// </value>
        public double[] GuessPosteriorMeans
        {
            get
            {
                return this.GuessPosteriors == null ? null : this.GuessPosteriors.GetMeans();
            }
        }



        /// <summary>
        /// Gets the is correct posterior means.
        /// </summary>
        /// <value>
        /// The is correct posterior means.
        /// </value>
        public double[][] IsCorrectPosteriorMeans
        {
            get
            {
                return this.IsCorrectPosteriors == null ? null : this.IsCorrectPosteriors.GetMeans();
            }
        }

        /// <summary>
        /// Gets the guess posterior means binned.
        /// </summary>
        /// <value>
        /// The guess posterior means binned.
        /// </value>
        public int[] GuessPosteriorMeansBinned
        {
            get
            {
                return this.GuessPosteriorMeans == null ? null : this.GuessPosteriorMeans.Bin(10, 0.0, 1.0);
            }
        }
    }
}
