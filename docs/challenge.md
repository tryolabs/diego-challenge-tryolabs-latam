

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

**Added constant variables**

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

**Code cleaning**

I also improved the efficiency and documentation of functions, take the `get_period_day` function as an example:

Before

```python
def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'
```

After

```python
def get_period_day(date: str) -> str:
    """
    Determine the period of the day (mañana, tarde, noche) based on the provided datetime string.

    Parameters
    ----------
    date : str
        A string representing the date and time in the format '%Y-%m-%d %H:%M:%S'.

    Returns
    -------
    str
        The period of the day:
        - 'mañana' (morning): 05:00 - 11:59
        - 'tarde' (afternoon): 12:00 - 18:59
        - 'noche' (night): 19:00 - 04:59
        If the date format is invalid, returns 'Invalid date format'.
    """
    try:
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        time_of_day = date_time.time()
        
        if time(5, 0) <= time_of_day < time(12, 0):
            return 'mañana'
        elif time(12, 0) <= time_of_day < time(19, 0):
            return 'tarde'
        else:
            return 'noche'
    except ValueError:
        return 'Invalid date format'
```

**Incorrect Delay Rate Calculation**


The delay rate (%) variable was calculated incorrectly. It was computed as the total number of flights divided by the number of delays, which results in a ratio, not a percentage.

For example, a delay rate of 19 for Houston means that for every 19 total flights, there is 1 delayed flight. This does not accurately represent a percentage measure. Therefore, I corrected the calculation to be the number of delays divided by the total number of flights, which provides a consistent percentage measure for the plots.