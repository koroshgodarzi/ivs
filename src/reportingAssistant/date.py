from datetime import date
from convertdate import islamic, hebrew, persian

def today_date(type= ["Persian"]):
    today = date.today()
    y, m, d = today.year, today.month, today.day

    output = {"Gregorian": today}
    if "Persian" in type:
        persian_date = persian.from_gregorian(y, m, d)
        output["Persian"] = persian_date
    if "Islamic" in type:
        islamic_date = islamic.from_gregorian(y, m, d)
        output["Islamic"] = islamic_date
    return output
