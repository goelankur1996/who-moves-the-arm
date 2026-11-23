# Scoring a trial

Score live right after the trial, then again blind from the video a day later (scored trials only).

## Success (1 or 0)

1 only if **all** of these hold at or before 120 s:
- the **named** object is entirely inside the **named** cup,
- the gripper is open,
- the arm is clear of the object.

Anything else is 0, including the right object in the wrong cup, or the object resting on the rim.

## Progress (0 to 4): the highest stage reached

| Score | Stage | Counts when |
|---|---|---|
| 0 | Nothing useful | The fingers never got within about 3 cm of the named object |
| 1 | Reached | The fingers got within about 3 cm of the named object, around it or right above it |
| 2 | Holding | The named object was lifted clear of the table in the gripper |
| 3 | Carried | The held object got over the opening of the named cup |
| 4 | Released inside | The object ended inside the named cup |

A trial that picks the **wrong** object scores 0 for progress.

## Failure tag: the first thing that went wrong

| Tag | Use when |
|---|---|
| perception | It went for the wrong object or the wrong cup, or misjudged where something was by more than a few cm |
| reach | It aimed at the right place but the fingers never got around the object (stopped short, landed beside it, knocked it away while approaching) |
| grasp | It got around the object but closed on nothing, pinched an edge, or the object slipped out while lifting |
| transport | It held the object but dropped it on the way, or the object hit a cup while travelling |
| release | It was over the named cup but the object landed outside, stuck to the fingers, or bounced out |
| freeze | It stopped making progress: time or call limit reached, the policy stalled, or the model gave up |
| none | Success |

Only the first failure counts. If it pinched an edge, the piece fell, and it then ran out of time, the tag is **grasp**.

## Notes

Anything a later reader would want: "retried the grasp twice", "camera timeout at 40 s", "arm brushed the green cup".

## Between trials

1. Wait until the arm is back at rest and the script has asked for the score.
2. Score.
3. Take the object out of the cup and set the table for the next trial (dry runs: anywhere in the grey band; scored trials: the next layout, placed with the ghost overlay).
