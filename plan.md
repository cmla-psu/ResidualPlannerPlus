# 10-Day Sprint Plan to Address Reviews

## Day 1-2: Quick Wins (Mathematical Corrections + Typos)
**Goal: Fix all mathematical/technical errors and typos**

### Day 1 Morning
- [ ] Fix Theorem 3: M_A(·;σ_A)² → M_A(·;σ_A²)
- [ ] Fix Example 2: Bold variable **v**
- [ ] Verify and fix Example 3: YR_A multiplication (check if -1 or -3 is correct)
- [ ] Fix abstract: Clarify matrix mechanism definition

### Day 1 Afternoon
- [ ] Fix Table 1: Correct closure(W_kload) definition
- [ ] Fix all 3 typos (p.2, p.5, p.9)
- [ ] Review all mathematical notation for consistency

### Day 2
- [ ] Add early clarification of "optimal" in introduction
- [ ] Add explicit statement: "optimal among Gaussian linear mechanisms"
- [ ] Highlight code release in introduction/experiments section

---

## Day 3-6: Critical Experiments (Parallel Tasks)
**Goal: Address Reviewer #1's main concern**

### Day 3-4: ResidualPlanner+ Experiments
- [ ] Run scalability benchmarks for ResidualPlanner+ with:
  - Range-marginal queries
  - Prefix-marginal queries
- [ ] Compare against HDMM scalability
- [ ] Generate timing data and memory usage

### Day 5: Discrete Gaussian Experiments
- [ ] Run utility experiments for discrete Gaussian approach
- [ ] Measure scalability impact
- [ ] Compare overhead vs non-hardened implementation

### Day 6: Single Kronecker Product Experiments (Reviewer #2)
- [ ] Test R×R×R×R×R workload
- [ ] Test P×P×P×P×P workload
- [ ] Compare ResidualPlanner+ vs HDMM
- [ ] Document which HDMM parameterization was used

---

## Day 7: Visualization + Presentation
**Goal: Make results accessible**

### Day 7 Morning
- [ ] Convert Tables 2-9 to graphs (line plots or bar charts)
- [ ] Keep original tables in appendix
- [ ] Make Figures 1, 2, 3 grayscale-friendly

### Day 7 Afternoon
- [ ] Add new experiment results (graphs + tables)
- [ ] Write captions and analysis for new figures

---

## Day 8: Major Writing Revisions
**Goal: Address Reviewer #3's presentation concerns**

### Day 8 Morning: Optimality Discussion
- [ ] Write paragraph explaining Gaussian linear mechanisms class
- [ ] Discuss relationship to data-independent noise mechanisms
- [ ] Justify focus on this mechanism class
- [ ] Reference prior work on matrix mechanism optimality

### Day 8 Afternoon: Restructuring
- [ ] Move Theorem 6 proof to appendix
- [ ] Streamline discretization section in main body
- [ ] Revise Section 7.1:
  - Add intuitive explanation of generalized marginals
  - Add intuitive descriptions to examples

---

## Day 9: Integration + Discussion
**Goal: Tie everything together**

### Day 9 Morning
- [ ] Integrate all new experimental results into text
- [ ] Update experimental section with new findings
- [ ] Add discussion of discrete noise generalizability (Reviewer #2's question)
- [ ] Add discussion of weight allocation for smaller marginals (Reviewer #2)

### Day 9 Afternoon
- [ ] Write responses to reviewers addressing each point
- [ ] Create a change log highlighting all modifications
- [ ] Ensure all cross-references are correct

---

## Day 10: Final Review + Polish
**Goal: Quality assurance**

### Day 10 Morning
- [ ] Proofread entire paper
- [ ] Check all figures render correctly
- [ ] Verify all mathematical corrections
- [ ] Check all citations and references

### Day 10 Afternoon
- [ ] Run LaTeX compilation multiple times
- [ ] Generate final PDF
- [ ] Double-check reviewer response letter
- [ ] Submit revision

