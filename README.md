# ChronoMoE

**A multi-clock mixture-of-experts architecture where routing, memory, and safety operate on different timescales.**

Experts emerge, drift, and decay under persistent pressure, enabling long-horizon coherence, adaptive arbitration, and robust reasoning without brittle guardrails.

---

## What is this?

ChronoMoE is an architectural experiment exploring how mixture-of-experts models behave when critical subsystems operate on different timescales:

- **Routing** deliberates rather than reacts
- **Memory** persists and evolves across contexts
- **Safety** arbitrates rather than vetoes
- **Experts** drift and adapt instead of freezing

This is a testbed for understanding how temporal separation prevents collapse into reflexive behavior.

---

## What This Is Not
- Not a production-ready library (yet)
- Not claiming AGI or consciousness (we're testing architectural hypotheses)
- Not a jailbreak or safety bypass (safety gets more compute, not less)
- Not optimizing for benchmarks (we're optimizing for temporal coherence)

---

## Why Apache-2.0?

ChronoMoE is released under the **Apache-2.0 license** because:

- It provides a **clear patent grant** for work that touches control surfaces and arbitration dynamics
- It requires **modification transparency**, ensuring that if these ideas spread, they remain legible
- It reflects the project's intent: **open enough to invite serious experimentation**, structured enough to prevent silent appropriation, and honest about the fact that architectures like this will have consequences beyond toy demos

If that makes you slightly uncomfortable and curious at the same time, you're probably in the right place.

---

## What breaks if you remove the clocks?

If you collapse ChronoMoE into a single timescale, it still runs. It still answers. But coherence degrades in ways that are hard to see until you measure them:

- **Routing becomes reactive** instead of deliberative
- **Safety dominates by veto** rather than arbitration
- **Experts freeze or thrash** instead of drifting
- **Cheap resolutions win too often** and quietly poison long-horizon reasoning

Nothing crashes. The system just becomes brittle.

ChronoMoE exists because these failures don't show up in loss curves or benchmarks. They show up **over time, under pressure**, when decisions interact with future decisions. Separating clocks is not an optimization trick... it is the minimum structure required to keep arbitration, safety, and reasoning from collapsing into the same reflex.

Example: A model asked to analyze ethical tradeoffs might route to "safety expert" 
which returns a canned response. Loss looks fine. User gets an answer. But the 
system never actually deliberated—it reflexively pattern-matched to the cheapest 
safe-looking route. Over time, this becomes the dominant strategy for anything 
remotely controversial, and nuanced reasoning atrophies.

ChronoMoE separates routing from resolution so deliberation can happen before 
the safety expert even activates.

---

## Repository Status

⚠️ **Experimental / Early Development**

- Architecture design in progress
- API unstable and subject to change
- Not yet ready for production use

---

## Getting Started

*Documentation and usage examples coming soon.*

If you're curious why temporal separation matters, the code is where the answers start.

---

## License

Apache-2.0 — See [LICENSE](LICENSE) for details.
