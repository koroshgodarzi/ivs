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


def get_sql_date_prompt(date_input: dict) -> str:
    """
    Parses Jalali dates from a dictionary, converts them to Gregorian, 
    and returns a string suitable for an LLM generating SQL.
    Safely handles None values for start or end dates.
    """
    try:
        start_str = None
        end_str = None
        
        # Safely extract and convert start_date if it exists
        start_dict = date_input.get('start_date')
        if start_dict and isinstance(start_dict, dict) and start_dict.get('Jalali'):
            start_jalali = start_dict['Jalali']
            s_year, s_month, s_day = map(int, start_jalali.split('/'))
            g_start = persian.to_gregorian(s_year, s_month, s_day)
            start_str = f"{g_start[0]:04d}-{g_start[1]:02d}-{g_start[2]:02d}"
            
        # Safely extract and convert end_date if it exists
        end_dict = date_input.get('end_date')
        if end_dict and isinstance(end_dict, dict) and end_dict.get('Jalali'):
            end_jalali = end_dict['Jalali']
            e_year, e_month, e_day = map(int, end_jalali.split('/'))
            g_end = persian.to_gregorian(e_year, e_month, e_day)
            end_str = f"{g_end[0]:04d}-{g_end[1]:02d}-{g_end[2]:02d}"

        # Construct a precise prompt for the LLM based on available dates
        if start_str and end_str:
            prompt = f"between '{start_str}' and '{end_str}' (inclusive)"
        elif start_str:
            prompt = f">= '{start_str}'"
        elif end_str:
            prompt = f"<= '{end_str}'"
        else:
            prompt = "without specific date constraints"
            
        return prompt

    except AttributeError:
        return "Error: Expected a string for Jalali date format."
    except ValueError:
        return "Error: Jalali dates must be in the 'YYYY/MM/DD' format."
    except Exception as e:
        # A catch-all for any other unpredictable input structures
        return f"Error: An unexpected error occurred: {e}"

# --- Example Usage ---
if __name__ == "__main__":
    dates = {
        'start_date': {'Jalali': '1405/02/20'}, 
        'end_date': {'Jalali': '1405/03/20'}
    }
    
    llm_prompt = get_sql_date_prompt(dates)
    print(llm_prompt)