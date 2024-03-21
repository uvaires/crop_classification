from datetime import datetime
import calendar


def get_band_dates(bands: list, interval_type='bi_monthly', year=2022, custom_intervals=None) -> dict:
    """
    Function to generate date intervals for temporal compositions of satellite data.

    Parameters:
        bands (list): List of bands of interest.
        interval_type (str): Type of interval to generate. Options: 'bi_monthly', 'quarterly', 'semester', 'custom',
         'annual'.
        Default is 'bi_monthly'.
        year (int): Year for which to generate the intervals. Default is 2022.
        custom_intervals (list of tuples): List of custom intervals. Each tuple should contain start and end months.
                                           Default is None.

    Returns:
        dict: Dictionary containing date ranges for the specified bands and intervals.
    """
    # Dictionary mapping month numbers to month names
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    # Dictionary to store date ranges
    date_ranges = {}

    # Loop through each band
    for band in bands:
        # Generate intervals based on the specified interval type
        if interval_type == 'bi_monthly':
            # Adding bi-monthly intervals
            for month in range(1, 13, 2):
                month_name_1 = month_names[month].lower()
                month_name_2 = month_names[min(month + 1, 12)].lower()
                start_date = datetime(year, month, 1)
                end_day = calendar.monthrange(year, min(month + 1, 12))[1]  # Last day of the month
                end_date = datetime(year, min(month + 1, 12), end_day)
                date_ranges[f'{band}_{month_name_1}_{month_name_2}'] = (start_date, end_date)
        elif interval_type == 'quarterly':
            # Adding quarterly intervals
            quarters = [(1, 3), (4, 6), (7, 9), (10, 12)]
            for quarter_num, (start_month, end_month) in enumerate(quarters, start=1):
                start_month_name = month_names[start_month]
                end_month_name = month_names[end_month]
                quarter_name = f'Q{quarter_num}_{start_month_name}_{end_month_name}'
                start_date = datetime(year, start_month, 1)
                end_day = calendar.monthrange(year, end_month)[1]  # Last day of the month
                end_date = datetime(year, end_month, end_day)
                date_ranges[f'{band}_{quarter_name}'] = (start_date, end_date)
        elif interval_type == 'semester':
            # Adding semester intervals
            date_ranges[f'{band}_semester1'] = (datetime(year, 1, 1), datetime(year, 6, 30))
            date_ranges[f'{band}_semester2'] = (datetime(year, 7, 1), datetime(year, 12, 31))
        elif interval_type == 'custom':
            # Adding custom intervals
            if custom_intervals:
                for index, (start_month, end_month) in enumerate(custom_intervals, start=1):
                    start_month_name = month_names[start_month]
                    end_month_name = month_names[end_month]
                    interval_name = f'{band}_{start_month_name.lower()}_{end_month_name.lower()}'
                    start_date = datetime(year, start_month, 1)
                    end_day = calendar.monthrange(year, end_month)[1]  # Last day of the month
                    end_date = datetime(year, end_month, end_day)
                    date_ranges[interval_name] = (start_date, end_date)
        elif interval_type == 'annual':
            # Adding an annual interval
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            date_ranges[f'{band}_annual'] = (start_date, end_date)

    return date_ranges
