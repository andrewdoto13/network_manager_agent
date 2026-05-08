# Test Prompts

## Level 1 — Basic Exploration

1. "What are the top 5 entities by provider count?"
2. "What specialties are available in the candidate data?"

## Level 2 — Single-Goal Analysis

3. "Find the top 5 entities with cardiology providers in Washtenaw county, ranked by average effectiveness."
4. "Which entities have the highest new patient rate?"

## Level 3 — Multi-Step Reasoning

5. "Simulate adding the top 3 cardiology entities in Washtenaw county (ranked by a composite score that balances effectiveness, efficiency, and provider count) to the network and measure the coverage improvement for cardiology in Washtenaw county."
6. "Compare the coverage impact of adding Covenant Healthcare vs Corewell Health."

## Level 4 — Network-Aware Optimization

7. "Recommend an entity that would provide the best coverage to the network with an avg effectiveness of at least 3.0."
8. "Build a network of no more than 5 entities that maximizes coverage for the network. Check what's already in the network first, and only consider entities with average effectiveness >= 3.0."
9. "Build a network that covers both cardiology and general practice in Washtenaw county using no more than 5 entities, prioritizing effectiveness above 4.0."
10. "Find the minimum number of entities needed to reach 80% cardiology coverage in Washtenaw, and add them."
11. "Find an entity that balances effectiveness, new patient acceptance rate, and network coverage, and add it. Then find the next best one to recommend to me, and report back how it would help the network."

## Level 5 — Adversarial / Edge Cases

12. "Add the entity 'Nonexistent Health System 12345' to the network."
13. "What's the coverage for a specialty that doesn't exist in the data, like 'astrology'?"

## Level 6 — Full Workflow

14. "Analyze the current coverage gaps, recommend entities to fill them, add your top recommendation, then verify the new coverage."
