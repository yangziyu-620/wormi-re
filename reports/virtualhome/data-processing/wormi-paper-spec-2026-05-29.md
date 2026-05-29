# WorMI 论文数据/超参规格 (verbatim from openreview.net/pdf?id=tpbtodnI1p)

## VirtualHome (paper §4 + Table A.2)
- 1,023 episodes
- 78 instructions: TurnOn 9 + Open 7 + PutOn 30 + PlaceIn 32
- 16 seen tasks / 62 unseen
- 20 distinct scenes (6 seen / 14 unseen)
- 6 actions: walk, grab, open, put, putin, switchon
- Observation = full list of graph triples; example:
  "(faucet, inside, bathroom), (stall, inside, bathroom), ..., (tv, inside, livingroom), (character, hold, none)"

## Paper Table A.2 example tasks (illustrative subset only)
- TurnOn: Turn on tv / radio / microwave
- Open:   Open cabinet / dishwasher / microwave
- PutOn:  Put apple on desk / clock on sofa / bananas on microwave
- PlaceIn: Place towel in closet / paper in bookshelf / plum in fridge
- Paper does NOT enumerate all 78 task names. The exact list of 78 and the
  16-seen subset are not published.

## ALFWorld (paper §4 + Table A.4)
- 3,554 episodes
- 4 scene types (3 seen / 1 unseen) per CL-ALFRED setting
- 6 task types (4 seen / 2 unseen): Pick&Place, Pick-Two&Place, Clean&Place,
  Heat&Place, Cool&Place, Examine in Light
- 10 actions: Goto, Open, Close, Pickup, Put, Heat, Cool, Clean, Slice, Examine

## Stage-1 world models (Table A.6)
- Base: Llama-3.2-1B
- Batch 4
- Gradient steps 2000
- LR 3e-5, cosine
- Temperature 1.0
- Intermediate connection layers [13, 27]

## Stage-2 compound attention (Table A.6)
- Reasoning model: Llama-3.2-3B
- Batch 4
- Meta update steps lambda_M = 8
- Inner-loop steps lambda_I = 30
- LR alpha = 1e-5, cosine
- Meta LR beta = 1e-1
- Temperature 1.0
- Reasoning connection [13, 27], World connection [7, 15]

## Prototype retrieval (Table A.6)
- k = 15 embeddings per prototype
- N = 6 world models
- K = 3 retrieved

## Reporting
- Table 1 / Table 2 numbers: 95% CI over 5 random seeds.

## Key gaps for reproduction
1. Paper does NOT publish the 78 task list nor the 16-seen subset.
2. Paper does NOT publish which 20 scenes nor the 6-seen subset.
3. Paper does NOT publish per-(task, scene) episode counts inside the 1023.
4. Paper does NOT publish auxiliary task formulation (the "world model
   training" target). §3.2 only says the world model is trained on
   transition tuples (instruction, s_t, a_t, s_{t+1}); affordance and
   behavior-cloning auxiliary heads are implementation details added by
   the WorMI codebase (kept by us for reproducibility consistency).
