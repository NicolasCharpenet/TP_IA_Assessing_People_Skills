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
	public partial class Model8_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>True if Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3 has executed. Set this to false to force re-execution of Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3</summary>
		public bool Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3_isDone;
		/// <summary>True if Changed_isCorrect0 has executed. Set this to false to force re-execution of Changed_isCorrect0</summary>
		public bool Changed_isCorrect0_isDone;
		/// <summary>True if Changed_isCorrect1 has executed. Set this to false to force re-execution of Changed_isCorrect1</summary>
		public bool Changed_isCorrect1_isDone;
		/// <summary>True if Changed_isCorrect2 has executed. Set this to false to force re-execution of Changed_isCorrect2</summary>
		public bool Changed_isCorrect2_isDone;
		/// <summary>True if Changed_isCorrect3 has executed. Set this to false to force re-execution of Changed_isCorrect3</summary>
		public bool Changed_isCorrect3_isDone;
		/// <summary>Message to marginal of 'csharp'</summary>
		public Bernoulli csharp_marginal_F;
		/// <summary>The constant 'hasSkills3_F'</summary>
		public bool hasSkills3_F;
		/// <summary>Message to marginal of 'hasSkills3_F'</summary>
		public Bernoulli hasSkills3_F_marginal_F;
		/// <summary>Message to marginal of 'hasSkills3_T'</summary>
		public Bernoulli hasSkills3_T_marginal_F;
		/// <summary>The constant 'hasSkills4_F'</summary>
		public bool hasSkills4_F;
		/// <summary>Message to marginal of 'hasSkills4_F'</summary>
		public Bernoulli hasSkills4_F_marginal_F;
		/// <summary>Message to marginal of 'hasSkills4_T'</summary>
		public Bernoulli hasSkills4_T_marginal_F;
		/// <summary>Field backing the isCorrect0 property</summary>
		private bool IsCorrect0;
		/// <summary>Message to marginal of 'isCorrect0'</summary>
		public Bernoulli isCorrect0_marginal_F;
		/// <summary>Field backing the isCorrect1 property</summary>
		private bool IsCorrect1;
		/// <summary>Message to marginal of 'isCorrect1'</summary>
		public Bernoulli isCorrect1_marginal_F;
		/// <summary>Field backing the isCorrect2 property</summary>
		private bool IsCorrect2;
		/// <summary>Message to marginal of 'isCorrect2'</summary>
		public Bernoulli isCorrect2_marginal_F;
		/// <summary>Field backing the isCorrect3 property</summary>
		private bool IsCorrect3;
		/// <summary>Message to marginal of 'isCorrect3'</summary>
		public Bernoulli isCorrect3_marginal_F;
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		/// <summary>Message to marginal of 'sql'</summary>
		public Bernoulli sql_marginal_F;
		#endregion

		#region Properties
		/// <summary>The externally-specified value of 'isCorrect0'</summary>
		public bool isCorrect0
		{
			get {
				return this.IsCorrect0;
			}
			set {
				if (this.IsCorrect0!=value) {
					this.IsCorrect0 = value;
					this.numberOfIterationsDone = 0;
					this.Changed_isCorrect0_isDone = false;
					this.Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3_isDone = false;
				}
			}
		}

		/// <summary>The externally-specified value of 'isCorrect1'</summary>
		public bool isCorrect1
		{
			get {
				return this.IsCorrect1;
			}
			set {
				if (this.IsCorrect1!=value) {
					this.IsCorrect1 = value;
					this.numberOfIterationsDone = 0;
					this.Changed_isCorrect1_isDone = false;
					this.Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3_isDone = false;
				}
			}
		}

		/// <summary>The externally-specified value of 'isCorrect2'</summary>
		public bool isCorrect2
		{
			get {
				return this.IsCorrect2;
			}
			set {
				if (this.IsCorrect2!=value) {
					this.IsCorrect2 = value;
					this.numberOfIterationsDone = 0;
					this.Changed_isCorrect2_isDone = false;
					this.Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3_isDone = false;
				}
			}
		}

		/// <summary>The externally-specified value of 'isCorrect3'</summary>
		public bool isCorrect3
		{
			get {
				return this.IsCorrect3;
			}
			set {
				if (this.IsCorrect3!=value) {
					this.IsCorrect3 = value;
					this.numberOfIterationsDone = 0;
					this.Changed_isCorrect3_isDone = false;
					this.Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3_isDone = false;
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
		/// <summary>Computations that depend on the observed value of isCorrect0</summary>
		private void Changed_isCorrect0()
		{
			if (this.Changed_isCorrect0_isDone) {
				return ;
			}
			this.isCorrect0_marginal_F = Bernoulli.Uniform();
			// Message to 'isCorrect0_marginal' from DerivedVariable factor
			this.isCorrect0_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.IsCorrect0, this.isCorrect0_marginal_F);
			this.Changed_isCorrect0_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of isCorrect0 and isCorrect1 and isCorrect2 and isCorrect3</summary>
		private void Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3()
		{
			if (this.Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3_isDone) {
				return ;
			}
			Bernoulli vBernoulli18 = Bernoulli.Uniform();
			this.csharp_marginal_F = Bernoulli.Uniform();
			Bernoulli[] csharp_selector_uses_B;
			// Create array for 'csharp_selector_uses' Backwards messages.
			csharp_selector_uses_B = new Bernoulli[2];
			csharp_selector_uses_B[1] = Bernoulli.Uniform();
			DistributionStructArray<Bernoulli,bool> csharp_selector_cases_B;
			// Create array for 'csharp_selector_cases' Backwards messages.
			csharp_selector_cases_B = new DistributionStructArray<Bernoulli,bool>(2);
			for(int _ind0 = 0; _ind0<2; _ind0++) {
				csharp_selector_cases_B[_ind0] = Bernoulli.Uniform();
			}
			Bernoulli[] csharp_selector_cases_0_uses_B;
			// Create array for 'csharp_selector_cases_0_uses' Backwards messages.
			csharp_selector_cases_0_uses_B = new Bernoulli[20];
			csharp_selector_cases_0_uses_B[18] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[17] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[16] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[15] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[14] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[12] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[11] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[10] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[9] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[8] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[6] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[5] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[3] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[2] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[1] = Bernoulli.Uniform();
			csharp_selector_cases_0_uses_B[0] = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_0_uses' from Bernoulli factor
			csharp_selector_cases_0_uses_B[0] = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect0, 0.90000000000000002));
			Bernoulli[] sql__0__uses_B;
			// Create array for 'sql__0__uses' Backwards messages.
			sql__0__uses_B = new Bernoulli[3];
			sql__0__uses_B[2] = Bernoulli.Uniform();
			sql__0__uses_B[1] = Bernoulli.Uniform();
			Bernoulli[] sql__0__uses_F;
			// Create array for 'sql__0__uses' Forwards messages.
			sql__0__uses_F = new Bernoulli[3];
			sql__0__uses_F[0] = Bernoulli.Uniform();
			DistributionStructArray<Bernoulli,bool> sql__0__selector_cases_B;
			// Create array for 'sql__0__selector_cases' Backwards messages.
			sql__0__selector_cases_B = new DistributionStructArray<Bernoulli,bool>(2);
			for(int _ind0 = 0; _ind0<2; _ind0++) {
				sql__0__selector_cases_B[_ind0] = Bernoulli.Uniform();
			}
			Bernoulli sql__0__selector_cases_0_B = default(Bernoulli);
			// Message to 'sql__0__selector_cases_0' from Bernoulli factor
			sql__0__selector_cases_0_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect1, 0.90000000000000002));
			// Message to 'sql__0__selector_cases' from Copy factor
			sql__0__selector_cases_B[0] = ArrayHelper.SetTo<Bernoulli>(sql__0__selector_cases_B[0], sql__0__selector_cases_0_B);
			Bernoulli sql__0__selector_cases_1_B = default(Bernoulli);
			// Message to 'sql__0__selector_cases_1' from Bernoulli factor
			sql__0__selector_cases_1_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect1, 0.20000000000000001));
			// Message to 'sql__0__selector_cases' from Copy factor
			sql__0__selector_cases_B[1] = ArrayHelper.SetTo<Bernoulli>(sql__0__selector_cases_B[1], sql__0__selector_cases_1_B);
			Bernoulli sql__0__selector_B = default(Bernoulli);
			// Message to 'sql__0__selector' from Cases factor
			sql__0__selector_B = CasesOp.BAverageConditional(sql__0__selector_cases_B);
			sql__0__uses_B[0] = Bernoulli.Uniform();
			// Message to 'sql__0__uses' from Copy factor
			sql__0__uses_B[0] = Tracing.FireEvent<Bernoulli>(ArrayHelper.SetTo<Bernoulli>(sql__0__uses_B[0], sql__0__selector_B), "sql__0__uses_B[0]", this.OnMessageUpdated, false);
			sql__0__uses_F[1] = Bernoulli.Uniform();
			// Message to 'sql__0__uses' from Replicate factor
			sql__0__uses_F[1] = Tracing.FireEvent<Bernoulli>(ReplicateOp_NoDivide.UsesAverageConditional<Bernoulli>(sql__0__uses_B, vBernoulli18, 1, sql__0__uses_F[1]), "sql__0__uses_F[1]", this.OnMessageUpdated, false);
			sql__0__uses_F[2] = Bernoulli.Uniform();
			// Message to 'sql__0__uses' from Replicate factor
			sql__0__uses_F[2] = Tracing.FireEvent<Bernoulli>(ReplicateOp_NoDivide.UsesAverageConditional<Bernoulli>(sql__0__uses_B, vBernoulli18, 2, sql__0__uses_F[2]), "sql__0__uses_F[2]", this.OnMessageUpdated, false);
			csharp_selector_cases_0_uses_B[4] = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_0_uses' from Replicate factor
			csharp_selector_cases_0_uses_B[4] = Bernoulli.FromLogOdds(ReplicateOp.LogEvidenceRatio<Bernoulli>(sql__0__uses_B, vBernoulli18, sql__0__uses_F));
			csharp_selector_cases_0_uses_B[7] = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_0_uses' from Cases factor
			csharp_selector_cases_0_uses_B[7] = Bernoulli.FromLogOdds(CasesOp.LogEvidenceRatio(sql__0__selector_cases_B, sql__0__uses_F[0]));
			Bernoulli hasSkills3_T_F = default(Bernoulli);
			hasSkills3_T_F = sql__0__uses_F[1];
			DistributionStructArray<Bernoulli,bool> hasSkills3_T_selector_cases_B;
			// Create array for 'hasSkills3_T_selector_cases' Backwards messages.
			hasSkills3_T_selector_cases_B = new DistributionStructArray<Bernoulli,bool>(2);
			for(int _ind0 = 0; _ind0<2; _ind0++) {
				hasSkills3_T_selector_cases_B[_ind0] = Bernoulli.Uniform();
			}
			Bernoulli hasSkills3_T_selector_cases_0_B = default(Bernoulli);
			// Message to 'hasSkills3_T_selector_cases_0' from Bernoulli factor
			hasSkills3_T_selector_cases_0_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect2, 0.90000000000000002));
			// Message to 'hasSkills3_T_selector_cases' from Copy factor
			hasSkills3_T_selector_cases_B[0] = ArrayHelper.SetTo<Bernoulli>(hasSkills3_T_selector_cases_B[0], hasSkills3_T_selector_cases_0_B);
			Bernoulli hasSkills3_T_selector_cases_1_B = default(Bernoulli);
			// Message to 'hasSkills3_T_selector_cases_1' from Bernoulli factor
			hasSkills3_T_selector_cases_1_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect2, 0.20000000000000001));
			// Message to 'hasSkills3_T_selector_cases' from Copy factor
			hasSkills3_T_selector_cases_B[1] = ArrayHelper.SetTo<Bernoulli>(hasSkills3_T_selector_cases_B[1], hasSkills3_T_selector_cases_1_B);
			csharp_selector_cases_0_uses_B[13] = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_0_uses' from Cases factor
			csharp_selector_cases_0_uses_B[13] = Bernoulli.FromLogOdds(CasesOp.LogEvidenceRatio(hasSkills3_T_selector_cases_B, hasSkills3_T_F));
			Bernoulli hasSkills4_T_F = default(Bernoulli);
			hasSkills4_T_F = sql__0__uses_F[2];
			DistributionStructArray<Bernoulli,bool> hasSkills4_T_selector_cases_B;
			// Create array for 'hasSkills4_T_selector_cases' Backwards messages.
			hasSkills4_T_selector_cases_B = new DistributionStructArray<Bernoulli,bool>(2);
			for(int _ind0 = 0; _ind0<2; _ind0++) {
				hasSkills4_T_selector_cases_B[_ind0] = Bernoulli.Uniform();
			}
			Bernoulli hasSkills4_T_selector_cases_0_B = default(Bernoulli);
			// Message to 'hasSkills4_T_selector_cases_0' from Bernoulli factor
			hasSkills4_T_selector_cases_0_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect3, 0.90000000000000002));
			// Message to 'hasSkills4_T_selector_cases' from Copy factor
			hasSkills4_T_selector_cases_B[0] = ArrayHelper.SetTo<Bernoulli>(hasSkills4_T_selector_cases_B[0], hasSkills4_T_selector_cases_0_B);
			Bernoulli hasSkills4_T_selector_cases_1_B = default(Bernoulli);
			// Message to 'hasSkills4_T_selector_cases_1' from Bernoulli factor
			hasSkills4_T_selector_cases_1_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect3, 0.20000000000000001));
			// Message to 'hasSkills4_T_selector_cases' from Copy factor
			hasSkills4_T_selector_cases_B[1] = ArrayHelper.SetTo<Bernoulli>(hasSkills4_T_selector_cases_B[1], hasSkills4_T_selector_cases_1_B);
			csharp_selector_cases_0_uses_B[19] = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_0_uses' from Cases factor
			csharp_selector_cases_0_uses_B[19] = Bernoulli.FromLogOdds(CasesOp.LogEvidenceRatio(hasSkills4_T_selector_cases_B, hasSkills4_T_F));
			Bernoulli csharp_selector_cases_0_B = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_0' from Replicate factor
			csharp_selector_cases_0_B = ReplicateOp_NoDivide.DefAverageConditional<Bernoulli>(csharp_selector_cases_0_uses_B, csharp_selector_cases_0_B);
			// Message to 'csharp_selector_cases' from Copy factor
			csharp_selector_cases_B[0] = ArrayHelper.SetTo<Bernoulli>(csharp_selector_cases_B[0], csharp_selector_cases_0_B);
			Bernoulli[] csharp_selector_cases_1_uses_B;
			// Create array for 'csharp_selector_cases_1_uses' Backwards messages.
			csharp_selector_cases_1_uses_B = new Bernoulli[8];
			csharp_selector_cases_1_uses_B[4] = Bernoulli.Uniform();
			csharp_selector_cases_1_uses_B[3] = Bernoulli.Uniform();
			csharp_selector_cases_1_uses_B[2] = Bernoulli.Uniform();
			csharp_selector_cases_1_uses_B[1] = Bernoulli.Uniform();
			csharp_selector_cases_1_uses_B[0] = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_1_uses' from Bernoulli factor
			csharp_selector_cases_1_uses_B[0] = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect0, 0.20000000000000001));
			DistributionStructArray<Bernoulli,bool> sql__1__selector_cases_B;
			// Create array for 'sql__1__selector_cases' Backwards messages.
			sql__1__selector_cases_B = new DistributionStructArray<Bernoulli,bool>(2);
			for(int _ind0 = 0; _ind0<2; _ind0++) {
				sql__1__selector_cases_B[_ind0] = Bernoulli.Uniform();
			}
			Bernoulli sql__1__selector_cases_0_B = default(Bernoulli);
			// Message to 'sql__1__selector_cases_0' from Bernoulli factor
			sql__1__selector_cases_0_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect1, 0.90000000000000002));
			// Message to 'sql__1__selector_cases' from Copy factor
			sql__1__selector_cases_B[0] = ArrayHelper.SetTo<Bernoulli>(sql__1__selector_cases_B[0], sql__1__selector_cases_0_B);
			Bernoulli sql__1__selector_cases_1_B = default(Bernoulli);
			// Message to 'sql__1__selector_cases_1' from Bernoulli factor
			sql__1__selector_cases_1_B = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect1, 0.20000000000000001));
			// Message to 'sql__1__selector_cases' from Copy factor
			sql__1__selector_cases_B[1] = ArrayHelper.SetTo<Bernoulli>(sql__1__selector_cases_B[1], sql__1__selector_cases_1_B);
			csharp_selector_cases_1_uses_B[5] = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_1_uses' from Cases factor
			csharp_selector_cases_1_uses_B[5] = Bernoulli.FromLogOdds(CasesOp.LogEvidenceRatio(sql__1__selector_cases_B, vBernoulli18));
			this.hasSkills3_F = false;
			csharp_selector_cases_1_uses_B[6] = Bernoulli.Uniform();
			if (this.hasSkills3_F) {
				// Message to 'csharp_selector_cases_1_uses' from Bernoulli factor
				csharp_selector_cases_1_uses_B[6] = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect2, 0.90000000000000002));
			}
			if (!this.hasSkills3_F) {
				// Message to 'csharp_selector_cases_1_uses' from Bernoulli factor
				csharp_selector_cases_1_uses_B[6] = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect2, 0.20000000000000001));
			}
			this.hasSkills4_F = false;
			csharp_selector_cases_1_uses_B[7] = Bernoulli.Uniform();
			if (this.hasSkills4_F) {
				// Message to 'csharp_selector_cases_1_uses' from Bernoulli factor
				csharp_selector_cases_1_uses_B[7] = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect3, 0.90000000000000002));
			}
			if (!this.hasSkills4_F) {
				// Message to 'csharp_selector_cases_1_uses' from Bernoulli factor
				csharp_selector_cases_1_uses_B[7] = Bernoulli.FromLogOdds(BernoulliFromBetaOp.LogEvidenceRatio(this.IsCorrect3, 0.20000000000000001));
			}
			Bernoulli csharp_selector_cases_1_B = Bernoulli.Uniform();
			// Message to 'csharp_selector_cases_1' from Replicate factor
			csharp_selector_cases_1_B = ReplicateOp_NoDivide.DefAverageConditional<Bernoulli>(csharp_selector_cases_1_uses_B, csharp_selector_cases_1_B);
			// Message to 'csharp_selector_cases' from Copy factor
			csharp_selector_cases_B[1] = ArrayHelper.SetTo<Bernoulli>(csharp_selector_cases_B[1], csharp_selector_cases_1_B);
			csharp_selector_uses_B[0] = Bernoulli.Uniform();
			// Message to 'csharp_selector_uses' from Cases factor
			csharp_selector_uses_B[0] = Tracing.FireEvent<Bernoulli>(CasesOp.BAverageConditional(csharp_selector_cases_B), "csharp_selector_uses_B[0]", this.OnMessageUpdated, false);
			Bernoulli csharp_selector_B = Bernoulli.Uniform();
			// Message to 'csharp_selector' from Replicate factor
			csharp_selector_B = ReplicateOp_NoDivide.DefAverageConditional<Bernoulli>(csharp_selector_uses_B, csharp_selector_B);
			// Message to 'csharp_marginal' from Variable factor
			this.csharp_marginal_F = VariableOp.MarginalAverageConditional<Bernoulli>(csharp_selector_B, vBernoulli18, this.csharp_marginal_F);
			this.sql_marginal_F = Bernoulli.Uniform();
			Bernoulli sql_use_B = Bernoulli.Uniform();
			Bernoulli[] csharp_selector_uses_F;
			// Create array for 'csharp_selector_uses' Forwards messages.
			csharp_selector_uses_F = new Bernoulli[2];
			csharp_selector_uses_F[1] = Bernoulli.Uniform();
			// Message to 'csharp_selector_uses' from Replicate factor
			csharp_selector_uses_F[1] = Tracing.FireEvent<Bernoulli>(ReplicateOp_NoDivide.UsesAverageConditional<Bernoulli>(csharp_selector_uses_B, vBernoulli18, 1, csharp_selector_uses_F[1]), "csharp_selector_uses_F[1]", this.OnMessageUpdated, false);
			Bernoulli[] sql__B;
			// Create array for 'sql_' Backwards messages.
			sql__B = new Bernoulli[2];
			for(int _gateind = 0; _gateind<2; _gateind++) {
				sql__B[_gateind] = Bernoulli.Uniform();
			}
			Bernoulli sql__0__B = Bernoulli.Uniform();
			// Message to 'sql__0_' from Replicate factor
			sql__0__B = ReplicateOp_NoDivide.DefAverageConditional<Bernoulli>(sql__0__uses_B, sql__0__B);
			// Message to 'sql_' from Copy factor
			sql__B[0] = ArrayHelper.SetTo<Bernoulli>(sql__B[0], sql__0__B);
			Bernoulli sql__1__selector_B = default(Bernoulli);
			// Message to 'sql__1__selector' from Cases factor
			sql__1__selector_B = CasesOp.BAverageConditional(sql__1__selector_cases_B);
			// Message to 'sql_' from Copy factor
			sql__B[1] = ArrayHelper.SetTo<Bernoulli>(sql__B[1], sql__1__selector_B);
			// Message to 'sql_use' from EnterPartial factor
			sql_use_B = BeliefPropagationGateEnterPartialOp.ValueAverageConditional<Bernoulli>(sql__B, csharp_selector_uses_F[1], vBernoulli18, new int[2] {0, 1}, sql_use_B);
			// Message to 'sql_marginal' from Variable factor
			this.sql_marginal_F = VariableOp.MarginalAverageConditional<Bernoulli>(sql_use_B, vBernoulli18, this.sql_marginal_F);
			this.hasSkills3_T_marginal_F = Bernoulli.Uniform();
			Bernoulli hasSkills3_T_selector_B = default(Bernoulli);
			// Message to 'hasSkills3_T_selector' from Cases factor
			hasSkills3_T_selector_B = CasesOp.BAverageConditional(hasSkills3_T_selector_cases_B);
			// Message to 'hasSkills3_T_marginal' from Variable factor
			this.hasSkills3_T_marginal_F = VariableOp.MarginalAverageConditional<Bernoulli>(hasSkills3_T_selector_B, hasSkills3_T_F, this.hasSkills3_T_marginal_F);
			this.hasSkills4_T_marginal_F = Bernoulli.Uniform();
			Bernoulli hasSkills4_T_selector_B = default(Bernoulli);
			// Message to 'hasSkills4_T_selector' from Cases factor
			hasSkills4_T_selector_B = CasesOp.BAverageConditional(hasSkills4_T_selector_cases_B);
			// Message to 'hasSkills4_T_marginal' from Variable factor
			this.hasSkills4_T_marginal_F = VariableOp.MarginalAverageConditional<Bernoulli>(hasSkills4_T_selector_B, hasSkills4_T_F, this.hasSkills4_T_marginal_F);
			this.hasSkills3_F_marginal_F = Bernoulli.Uniform();
			// Message to 'hasSkills3_F_marginal' from DerivedVariable factor
			this.hasSkills3_F_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.hasSkills3_F, this.hasSkills3_F_marginal_F);
			this.hasSkills4_F_marginal_F = Bernoulli.Uniform();
			// Message to 'hasSkills4_F_marginal' from DerivedVariable factor
			this.hasSkills4_F_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.hasSkills4_F, this.hasSkills4_F_marginal_F);
			this.Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of isCorrect1</summary>
		private void Changed_isCorrect1()
		{
			if (this.Changed_isCorrect1_isDone) {
				return ;
			}
			this.isCorrect1_marginal_F = Bernoulli.Uniform();
			// Message to 'isCorrect1_marginal' from DerivedVariable factor
			this.isCorrect1_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.IsCorrect1, this.isCorrect1_marginal_F);
			this.Changed_isCorrect1_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of isCorrect2</summary>
		private void Changed_isCorrect2()
		{
			if (this.Changed_isCorrect2_isDone) {
				return ;
			}
			this.isCorrect2_marginal_F = Bernoulli.Uniform();
			// Message to 'isCorrect2_marginal' from DerivedVariable factor
			this.isCorrect2_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.IsCorrect2, this.isCorrect2_marginal_F);
			this.Changed_isCorrect2_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of isCorrect3</summary>
		private void Changed_isCorrect3()
		{
			if (this.Changed_isCorrect3_isDone) {
				return ;
			}
			this.isCorrect3_marginal_F = Bernoulli.Uniform();
			// Message to 'isCorrect3_marginal' from DerivedVariable factor
			this.isCorrect3_marginal_F = DerivedVariableOp.MarginalAverageConditional<Bernoulli,bool>(this.IsCorrect3, this.isCorrect3_marginal_F);
			this.Changed_isCorrect3_isDone = true;
		}

		/// <summary>
		/// Returns the marginal distribution for 'csharp' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli CsharpMarginal()
		{
			return this.csharp_marginal_F;
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			this.Changed_isCorrect3();
			this.Changed_isCorrect2();
			this.Changed_isCorrect1();
			this.Changed_isCorrect0();
			this.Changed_isCorrect0_isCorrect1_isCorrect2_isCorrect3();
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
			if (variableName=="isCorrect0") {
				return this.isCorrect0;
			}
			if (variableName=="isCorrect1") {
				return this.isCorrect1;
			}
			if (variableName=="isCorrect2") {
				return this.isCorrect2;
			}
			if (variableName=="isCorrect3") {
				return this.isCorrect3;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>
		/// Returns the marginal distribution for 'hasSkills3_F' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli HasSkills3_FMarginal()
		{
			return this.hasSkills3_F_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'hasSkills3_T' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli HasSkills3_TMarginal()
		{
			return this.hasSkills3_T_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'hasSkills4_F' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli HasSkills4_FMarginal()
		{
			return this.hasSkills4_F_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'hasSkills4_T' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli HasSkills4_TMarginal()
		{
			return this.hasSkills4_T_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'isCorrect0' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli IsCorrect0Marginal()
		{
			return this.isCorrect0_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'isCorrect1' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli IsCorrect1Marginal()
		{
			return this.isCorrect1_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'isCorrect2' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli IsCorrect2Marginal()
		{
			return this.isCorrect2_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'isCorrect3' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli IsCorrect3Marginal()
		{
			return this.isCorrect3_marginal_F;
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="isCorrect3") {
				return this.IsCorrect3Marginal();
			}
			if (variableName=="isCorrect2") {
				return this.IsCorrect2Marginal();
			}
			if (variableName=="isCorrect1") {
				return this.IsCorrect1Marginal();
			}
			if (variableName=="isCorrect0") {
				return this.IsCorrect0Marginal();
			}
			if (variableName=="csharp") {
				return this.CsharpMarginal();
			}
			if (variableName=="sql") {
				return this.SqlMarginal();
			}
			if (variableName=="hasSkills3_T") {
				return this.HasSkills3_TMarginal();
			}
			if (variableName=="hasSkills4_T") {
				return this.HasSkills4_TMarginal();
			}
			if (variableName=="hasSkills3_F") {
				return this.HasSkills3_FMarginal();
			}
			if (variableName=="hasSkills4_F") {
				return this.HasSkills4_FMarginal();
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
			if (variableName=="isCorrect0") {
				this.isCorrect0 = (bool)value;
				return ;
			}
			if (variableName=="isCorrect1") {
				this.isCorrect1 = (bool)value;
				return ;
			}
			if (variableName=="isCorrect2") {
				this.isCorrect2 = (bool)value;
				return ;
			}
			if (variableName=="isCorrect3") {
				this.isCorrect3 = (bool)value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>
		/// Returns the marginal distribution for 'sql' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli SqlMarginal()
		{
			return this.sql_marginal_F;
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
