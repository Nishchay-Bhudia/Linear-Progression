# Linear Regression From Scratch

One Python script that fits a straight line to numbers you type in, using gradient
descent written out by hand in NumPy. No scikit-learn, no `fit()` call. I wrote it to
see the maths run rather than to have a regression tool.

## What gradient descent is doing

The line has two numbers that can change, a slope and an intercept. Start with both at
zero and the line is flat and wrong, and you can measure exactly how wrong by squaring
the gap between each predicted y and the real y. Calculus tells you which direction to
nudge the slope and the intercept to make that error smaller, so the script nudges them
a tiny amount in that direction and measures again. Do that a thousand times and the
line walks itself into place.

## Running it

```bash
pip install numpy matplotlib
python "linear progession.py"
```

It asks for your x values, then your y values, both comma separated. After training it
prints the learned slope and intercept, asks for one new x, and prints the predicted y.
Two matplotlib windows follow: the data with the fitted line and the prediction marked,
then the loss over the 1000 training steps. That second plot is the useful one. If the
loss is still falling steeply at the right hand edge, training stopped too early.

## Current state

It works, with rough edges I have left in on purpose so the script stays readable:

- The learning rate is hardcoded at 0.000001, which is very cautious. On small numbers
  the line barely moves in 1000 epochs. Raise it and watch the loss curve, that trade
  between a rate too small to converge and one large enough to diverge is most of what
  the project taught me.
- No input validation. Give it x and y lists of different lengths and NumPy will throw.
- Single feature only, one x per y.
- The script name is misspelled ("progession"), which I have kept so old links still work.

## Tech

Python, NumPy, matplotlib. MIT licensed.
