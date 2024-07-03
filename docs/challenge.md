

## Notebook `.ipnyb` improvements

### 1 Fix plotting issues

The function `sns.barplot()` expects positional arguments. I had to add the `x=` and `y=` positional arguments to fix the plots.

### 2 Improve plots

#### 2.1 Visual Improvements

There were plots where the x axis labels were wrongly rotated:

Before

![image](/docs/images/not_rotated_xlabels.png)

After

![image](/docs/images/rotated_xlabels.png)

Also, there were plots where the there are several x values and it would have been better to rotate the plot for better analysis. For example:

Before

![image](/docs/images/not_rotated_plot.png)

After

![image](/docs/images/rotated_plot.png)

Some plots were not rotated nor sorted:

Before

![image](/docs/images/missing_sorting.png)

After

![image](/docs/images/sorted.png)

The days of the week were not sorted and were in Spanish in the Delay Rate by Day of the Week plot:

Before

![image](/docs/images/days_not_sorted.png)


After

![image](/docs/images/days_sorted.png)

> [!NOTE] 
> More improvements were made on the notebook, but I decided to not document every improvement. For the full version, check the notebook present in this GitHub repository.

#### 2.2 Code improvements

It is not necessary to declare the font size in every plot, it can be once it the beginning.

```python
# Set the default font size
FONT_SIZE = 12
plt.rcParams.update({'font.size': FONT_SIZE})
```

The same can be done with the seaborn theme:

```python
sns.set_theme(style="darkgrid")
```