// <auto-generated />
#pragma warning disable 1570, 1591

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Collections;

namespace Models
{
	/// <summary>
	/// Generated algorithm for performing inference.
	/// </summary>
	/// <remarks>
	/// If you wish to use this class directly, you must perform the following steps:
	/// 1) Create an instance of the class.
	/// 2) Set the value of any externally-set fields e.g. data, priors.
	/// 3) Call the Execute(numberOfIterations) method.
	/// 4) Use the XXXMarginal() methods to retrieve posterior marginals for different variables.
	/// 
	/// Generated by Infer.NET 0.3.2102.1701 at 02:37 on vendredi 26 novembre 2021.
	/// </remarks>
	public partial class Model4_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>True if Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5 has executed. Set this to false to force re-execution of Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5</summary>
		public bool Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5_isDone;
		/// <summary>True if Changed_isCorrect0Minus5 has executed. Set this to false to force re-execution of Changed_isCorrect0Minus5</summary>
		public bool Changed_isCorrect0Minus5_isDone;
		/// <summary>True if Changed_isCorrect1Minus5 has executed. Set this to false to force re-execution of Changed_isCorrect1Minus5</summary>
		public bool Changed_isCorrect1Minus5_isDone;
		/// <summary>True if Changed_isCorrect2Minus5 has executed. Set this to false to force re-execution of Changed_isCorrect2Minus5</summary>
		public bool Changed_isCorrect2Minus5_isDone;
		/// <summary>Message to marginal of 'csharpMinus5'</summary>
		public Bernoulli csharpMinus5_marginal_F;
		/// <summary>Message to marginal of 'hasSkills3'</summary>
		public Bernoulli hasSkills3_marginal_F;
		/// <summary>Field backing the isCorrect0Minus5 property</summary>
		private bool IsCorrect0Minus5;
		/// <summary>Message to marginal of 'isCorrect0Minus5'</summary>
		public Bernoulli isCorrect0Minus5_marginal_F;
		/// <summary>Field backing the isCorrect1Minus5 property</summary>
		private bool IsCorrect1Minus5;
		/// <summary>Message to marginal of 'isCorrect1Minus5'</summary>
		public Bernoulli isCorrect1Minus5_marginal_F;
		/// <summary>Field backing the isCorrect2Minus5 property</summary>
		private bool IsCorrect2Minus5;
		/// <summary>Message to marginal of 'isCorrect2Minus5'</summary>
		public Bernoulli isCorrect2Minus5_marginal_F;
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		/// <summary>Message to marginal of 'sqlMinus5'</summary>
		public Bernoulli sqlMinus5_marginal_F;
		#endregion

		#region Properties
		/// <summary>The externally-specified value of 'isCorrect0Minus5'</summary>
		public bool isCorrect0Minus5
		{
			get {
				return this.IsCorrect0Minus5;
			}
			set {
				if (this.IsCorrect0Minus5!=value) {
					this.IsCorrect0Minus5 = value;
					this.numberOfIterationsDone = 0;
					this.Changed_isCorrect0Minus5_isDone = false;
					this.Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5_isDone = false;
				}
			}
		}

		/// <summary>The externally-specified value of 'isCorrect1Minus5'</summary>
		public bool isCorrect1Minus5
		{
			get {
				return this.IsCorrect1Minus5;
			}
			set {
				if (this.IsCorrect1Minus5!=value) {
					this.IsCorrect1Minus5 = value;
					this.numberOfIterationsDone = 0;
					this.Changed_isCorrect1Minus5_isDone = false;
					this.Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5_isDone = false;
				}
			}
		}

		/// <summary>The externally-specified value of 'isCorrect2Minus5'</summary>
		public bool isCorrect2Minus5
		{
			get {
				return this.IsCorrect2Minus5;
			}
			set {
				if (this.IsCorrect2Minus5!=value) {
					this.IsCorrect2Minus5 = value;
					this.numberOfIterationsDone = 0;
					this.Changed_isCorrect2Minus5_isDone = false;
					this.Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5_isDone = false;
				}
			}
		}

		/// <summary>The number of iterations done from the initial state</summary>
		public int NumberOfIterationsDone
		{
			get {
				return this.numberOfIterationsDone;
			}
		}

		#endregion

		#region Methods
		/// <summary>Computations that depend on the observed value of isCorrect0Minus5</summary>
		private void Changed_isCorrect0Minus5()
		{
			if (this.Changed_isCorrect0Minus5_isDone) {
				return ;
			}
			this.isCorrect0Minus5_marginal_F = Bernoulli.Uniform();
			// Message to 'isCorrect0Minus5_marginal' from DerivedVariable factor
			this.isCorrect0Minus5_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.IsCorrect0Minus5, this.isCorrect0Minus5_marginal_F);
			this.Changed_isCorrect0Minus5_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of isCorrect0Minus5 and isCorrect1Minus5 and isCorrect2Minus5</summary>
		private void Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5()
		{
			if (this.Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5_isDone) {
				return ;
			}
			Bernoulli vBernoulli10 = Bernoulli.Uniform();
			this.csharpMinus5_marginal_F = Bernoulli.Uniform();
			Bernoulli csharpMinus5_use_B = Bernoulli.Uniform();
			Bernoulli[] csharpMinus5_uses_F;
			Bernoulli[] csharpMinus5_uses_B;
			// Create array for 'csharpMinus5_uses' Forwards messages.
			csharpMinus5_uses_F = new Bernoulli[2];
			// Create array for 'csharpMinus5_uses' Backwards messages.
			csharpMinus5_uses_B = new Bernoulli[2];
			csharpMinus5_uses_B[1] = Bernoulli.Uniform();
			csharpMinus5_uses_B[0] = Bernoulli.Uniform();
			csharpMinus5_uses_F[1] = Bernoulli.Uniform();
			DistributionStructArray<Bernoulli,bool> csharpMinus5_selector_cases_B;
			// Create array for 'csharpMinus5_selector_cases' Backwards messages.
			csharpMinus5_selector_cases_B = new DistributionStructArray<Bernoulli,bool>(2);
			for(int _ind0 = 0; _ind0<2; _ind0++) {
				csharpMinus5_selector_cases_B[_ind0] = Bernoulli.Uniform();
			}
			Bernoulli csharpMinus5_selector_cases_0_B = default(Bernoulli);
			// Message to 'csharpMinus5_selector_cases_0' from Bernoulli factor
			csharpMinus5_selector_cases_0_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect0Minus5, 0.90000000000000002));
			// Message to 'csharpMinus5_selector_cases' from Copy factor
			csharpMinus5_selector_cases_B[0] = ArrayHelper.SetTo<Bernoulli>(csharpMinus5_selector_cases_B[0], csharpMinus5_selector_cases_0_B);
			Bernoulli csharpMinus5_selector_cases_1_B = default(Bernoulli);
			// Message to 'csharpMinus5_selector_cases_1' from Bernoulli factor
			csharpMinus5_selector_cases_1_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect0Minus5, 0.20000000000000001));
			// Message to 'csharpMinus5_selector_cases' from Copy factor
			csharpMinus5_selector_cases_B[1] = ArrayHelper.SetTo<Bernoulli>(csharpMinus5_selector_cases_B[1], csharpMinus5_selector_cases_1_B);
			Bernoulli csharpMinus5_selector_B = default(Bernoulli);
			// Message to 'csharpMinus5_selector' from Cases factor
			csharpMinus5_selector_B = CasesOp.BAverageConditional(csharpMinus5_selector_cases_B);
			// Message to 'csharpMinus5_uses' from Copy factor
			csharpMinus5_uses_B[0] = Tracing.FireEvent<Bernoulli>(ArrayHelper.SetTo<Bernoulli>(csharpMinus5_uses_B[0], csharpMinus5_selector_B), "csharpMinus5_uses_B[0]", this.OnMessageUpdated, false);
			DistributionStructArray<Bernoulli,bool> sqlMinus5_selector_cases_B;
			// Create array for 'sqlMinus5_selector_cases' Backwards messages.
			sqlMinus5_selector_cases_B = new DistributionStructArray<Bernoulli,bool>(2);
			for(int _ind0 = 0; _ind0<2; _ind0++) {
				sqlMinus5_selector_cases_B[_ind0] = Bernoulli.Uniform();
			}
			Bernoulli sqlMinus5_selector_cases_0_B = default(Bernoulli);
			// Message to 'sqlMinus5_selector_cases_0' from Bernoulli factor
			sqlMinus5_selector_cases_0_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect1Minus5, 0.90000000000000002));
			// Message to 'sqlMinus5_selector_cases' from Copy factor
			sqlMinus5_selector_cases_B[0] = ArrayHelper.SetTo<Bernoulli>(sqlMinus5_selector_cases_B[0], sqlMinus5_selector_cases_0_B);
			Bernoulli sqlMinus5_selector_cases_1_B = default(Bernoulli);
			// Message to 'sqlMinus5_selector_cases_1' from Bernoulli factor
			sqlMinus5_selector_cases_1_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect1Minus5, 0.20000000000000001));
			// Message to 'sqlMinus5_selector_cases' from Copy factor
			sqlMinus5_selector_cases_B[1] = ArrayHelper.SetTo<Bernoulli>(sqlMinus5_selector_cases_B[1], sqlMinus5_selector_cases_1_B);
			Bernoulli sqlMinus5_selector_B = default(Bernoulli);
			// Message to 'sqlMinus5_selector' from Cases factor
			sqlMinus5_selector_B = CasesOp.BAverageConditional(sqlMinus5_selector_cases_B);
			Bernoulli[] sqlMinus5_uses_B;
			// Create array for 'sqlMinus5_uses' Backwards messages.
			sqlMinus5_uses_B = new Bernoulli[2];
			sqlMinus5_uses_B[0] = Bernoulli.Uniform();
			// Message to 'sqlMinus5_uses' from Copy factor
			sqlMinus5_uses_B[0] = Tracing.FireEvent<Bernoulli>(ArrayHelper.SetTo<Bernoulli>(sqlMinus5_uses_B[0], sqlMinus5_selector_B), "sqlMinus5_uses_B[0]", this.OnMessageUpdated, false);
			Bernoulli[] sqlMinus5_uses_F;
			// Create array for 'sqlMinus5_uses' Forwards messages.
			sqlMinus5_uses_F = new Bernoulli[2];
			sqlMinus5_uses_F[1] = Bernoulli.Uniform();
			// Message to 'sqlMinus5_uses' from Replicate factor
			sqlMinus5_uses_F[1] = Tracing.FireEvent<Bernoulli>(ReplicateOp_NoDivide.UsesAverageConditional<Bernoulli>(sqlMinus5_uses_B, vBernoulli10, 1, sqlMinus5_uses_F[1]), "sqlMinus5_uses_F[1]", this.OnMessageUpdated, false);
			DistributionStructArray<Bernoulli,bool> hasSkills3_selector_cases_B;
			// Create array for 'hasSkills3_selector_cases' Backwards messages.
			hasSkills3_selector_cases_B = new DistributionStructArray<Bernoulli,bool>(2);
			for(int _ind0 = 0; _ind0<2; _ind0++) {
				hasSkills3_selector_cases_B[_ind0] = Bernoulli.Uniform();
			}
			Bernoulli hasSkills3_selector_cases_0_B = default(Bernoulli);
			// Message to 'hasSkills3_selector_cases_0' from Bernoulli factor
			hasSkills3_selector_cases_0_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect2Minus5, 0.90000000000000002));
			// Message to 'hasSkills3_selector_cases' from Copy factor
			hasSkills3_selector_cases_B[0] = ArrayHelper.SetTo<Bernoulli>(hasSkills3_selector_cases_B[0], hasSkills3_selector_cases_0_B);
			Bernoulli hasSkills3_selector_cases_1_B = default(Bernoulli);
			// Message to 'hasSkills3_selector_cases_1' from Bernoulli factor
			hasSkills3_selector_cases_1_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect2Minus5, 0.20000000000000001));
			// Message to 'hasSkills3_selector_cases' from Copy factor
			hasSkills3_selector_cases_B[1] = ArrayHelper.SetTo<Bernoulli>(hasSkills3_selector_cases_B[1], hasSkills3_selector_cases_1_B);
			Bernoulli hasSkills3_selector_B = default(Bernoulli);
			// Message to 'hasSkills3_selector' from Cases factor
			hasSkills3_selector_B = CasesOp.BAverageConditional(hasSkills3_selector_cases_B);
			// Message to 'csharpMinus5_uses' from And factor
			csharpMinus5_uses_B[1] = Tracing.FireEvent<Bernoulli>(BooleanAndOp.AAverageConditional(hasSkills3_selector_B, sqlMinus5_uses_F[1]), "csharpMinus5_uses_B[1]", this.OnMessageUpdated, false);
			// Message to 'csharpMinus5_use' from Replicate factor
			csharpMinus5_use_B = ReplicateOp_NoDivide.DefAverageConditional<Bernoulli>(csharpMinus5_uses_B, csharpMinus5_use_B);
			// Message to 'csharpMinus5_marginal' from Variable factor
			this.csharpMinus5_marginal_F = VariableOp.MarginalAverageConditional<Bernoulli>(csharpMinus5_use_B, vBernoulli10, this.csharpMinus5_marginal_F);
			// Message to 'csharpMinus5_uses' from Replicate factor
			csharpMinus5_uses_F[1] = Tracing.FireEvent<Bernoulli>(ReplicateOp_NoDivide.UsesAverageConditional<Bernoulli>(csharpMinus5_uses_B, vBernoulli10, 1, csharpMinus5_uses_F[1]), "csharpMinus5_uses_F[1]", this.OnMessageUpdated, false);
			this.sqlMinus5_marginal_F = Bernoulli.Uniform();
			Bernoulli sqlMinus5_use_B = Bernoulli.Uniform();
			sqlMinus5_uses_B[1] = Bernoulli.Uniform();
			// Message to 'sqlMinus5_uses' from And factor
			sqlMinus5_uses_B[1] = Tracing.FireEvent<Bernoulli>(BooleanAndOp.BAverageConditional(hasSkills3_selector_B, csharpMinus5_uses_F[1]), "sqlMinus5_uses_B[1]", this.OnMessageUpdated, false);
			// Message to 'sqlMinus5_use' from Replicate factor
			sqlMinus5_use_B = ReplicateOp_NoDivide.DefAverageConditional<Bernoulli>(sqlMinus5_uses_B, sqlMinus5_use_B);
			// Message to 'sqlMinus5_marginal' from Variable factor
			this.sqlMinus5_marginal_F = VariableOp.MarginalAverageConditional<Bernoulli>(sqlMinus5_use_B, vBernoulli10, this.sqlMinus5_marginal_F);
			Bernoulli hasSkills3_F = default(Bernoulli);
			this.hasSkills3_marginal_F = Bernoulli.Uniform();
			// Message to 'hasSkills3' from And factor
			hasSkills3_F = BooleanAndOp.AndAverageConditional(csharpMinus5_uses_F[1], sqlMinus5_uses_F[1]);
			// Message to 'hasSkills3_marginal' from DerivedVariable factor
			this.hasSkills3_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli>(hasSkills3_selector_B, hasSkills3_F, this.hasSkills3_marginal_F);
			this.Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of isCorrect1Minus5</summary>
		private void Changed_isCorrect1Minus5()
		{
			if (this.Changed_isCorrect1Minus5_isDone) {
				return ;
			}
			this.isCorrect1Minus5_marginal_F = Bernoulli.Uniform();
			// Message to 'isCorrect1Minus5_marginal' from DerivedVariable factor
			this.isCorrect1Minus5_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.IsCorrect1Minus5, this.isCorrect1Minus5_marginal_F);
			this.Changed_isCorrect1Minus5_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of isCorrect2Minus5</summary>
		private void Changed_isCorrect2Minus5()
		{
			if (this.Changed_isCorrect2Minus5_isDone) {
				return ;
			}
			this.isCorrect2Minus5_marginal_F = Bernoulli.Uniform();
			// Message to 'isCorrect2Minus5_marginal' from DerivedVariable factor
			this.isCorrect2Minus5_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.IsCorrect2Minus5, this.isCorrect2Minus5_marginal_F);
			this.Changed_isCorrect2Minus5_isDone = true;
		}

		/// <summary>
		/// Returns the marginal distribution for 'csharpMinus5' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli CsharpMinus5Marginal()
		{
			return this.csharpMinus5_marginal_F;
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			this.Changed_isCorrect2Minus5();
			this.Changed_isCorrect1Minus5();
			this.Changed_isCorrect0Minus5();
			this.Changed_isCorrect0Minus5_isCorrect1Minus5_isCorrect2Minus5();
			this.numberOfIterationsDone = numberOfIterations;
		}

		/// <summary>Update all marginals, by iterating message-passing the given number of times</summary>
		/// <param name="numberOfIterations">The total number of iterations that should be executed for the current set of observed values.  If this is more than the number already done, only the extra iterations are done.  If this is less than the number already done, message-passing is restarted from the beginning.  Changing the observed values resets the iteration count to 0.</param>
		public void Execute(int numberOfIterations)
		{
			this.Execute(numberOfIterations, true);
		}

		/// <summary>Get the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		public object GetObservedValue(string variableName)
		{
			if (variableName=="isCorrect0Minus5") {
				return this.isCorrect0Minus5;
			}
			if (variableName=="isCorrect1Minus5") {
				return this.isCorrect1Minus5;
			}
			if (variableName=="isCorrect2Minus5") {
				return this.isCorrect2Minus5;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>
		/// Returns the marginal distribution for 'hasSkills3' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli HasSkills3Marginal()
		{
			return this.hasSkills3_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'isCorrect0Minus5' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli IsCorrect0Minus5Marginal()
		{
			return this.isCorrect0Minus5_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'isCorrect1Minus5' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli IsCorrect1Minus5Marginal()
		{
			return this.isCorrect1Minus5_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'isCorrect2Minus5' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli IsCorrect2Minus5Marginal()
		{
			return this.isCorrect2Minus5_marginal_F;
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="isCorrect2Minus5") {
				return this.IsCorrect2Minus5Marginal();
			}
			if (variableName=="isCorrect1Minus5") {
				return this.IsCorrect1Minus5Marginal();
			}
			if (variableName=="isCorrect0Minus5") {
				return this.IsCorrect0Minus5Marginal();
			}
			if (variableName=="csharpMinus5") {
				return this.CsharpMinus5Marginal();
			}
			if (variableName=="sqlMinus5") {
				return this.SqlMinus5Marginal();
			}
			if (variableName=="hasSkills3") {
				return this.HasSkills3Marginal();
			}
			throw new ArgumentException("This class was not built to infer "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName));
		}

		/// <summary>Get the query-specific marginal distribution of a variable.</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName, string query)
		{
			if (query=="Marginal") {
				return this.Marginal(variableName);
			}
			throw new ArgumentException(((("This class was not built to infer \'"+variableName)+"\' with query \'")+query)+"\'");
		}

		/// <summary>Get the query-specific marginal distribution of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName, string query)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName, query));
		}

		private void OnMessageUpdated(MessageUpdatedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<MessageUpdatedEventArgs> handler = this.MessageUpdated;
			if (handler!=null) {
				handler(this, e);
			}
		}

		private void OnProgressChanged(ProgressChangedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<ProgressChangedEventArgs> handler = this.ProgressChanged;
			if (handler!=null) {
				handler(this, e);
			}
		}

		/// <summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>
		public void Reset()
		{
			this.Execute(0);
		}

		/// <summary>Set the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		/// <param name="value">Observed value</param>
		public void SetObservedValue(string variableName, object value)
		{
			if (variableName=="isCorrect0Minus5") {
				this.isCorrect0Minus5 = (bool)value;
				return ;
			}
			if (variableName=="isCorrect1Minus5") {
				this.isCorrect1Minus5 = (bool)value;
				return ;
			}
			if (variableName=="isCorrect2Minus5") {
				this.isCorrect2Minus5 = (bool)value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>
		/// Returns the marginal distribution for 'sqlMinus5' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli SqlMinus5Marginal()
		{
			return this.sqlMinus5_marginal_F;
		}

		/// <summary>Update all marginals, by iterating message-passing an additional number of times</summary>
		/// <param name="additionalIterations">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>
		public void Update(int additionalIterations)
		{
			this.Execute(checked(this.numberOfIterationsDone+additionalIterations), false);
		}

		#endregion

		#region Events
		/// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
		public event EventHandler<ProgressChangedEventArgs> ProgressChanged;
		/// <summary>Event that is fired when a message that is being monitored is updated.</summary>
		public event EventHandler<MessageUpdatedEventArgs> MessageUpdated;
		#endregion

	}

}
