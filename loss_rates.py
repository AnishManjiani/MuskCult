import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import constants
from data_mapping_final import convert_date
import datetime


class LossRateMatrices:
    """
    Computes loss rates at maturity for each Term, Grade and Status e.g.
    from utils.loss_rates import LossRateMatrices
    loss_rates = LossRateMatrices(df)
    date =  pd.to_datetime('2015-06-01T00:00:00.000000000').to_pydatetime()
    df_lr = loss_rates.generate_matrices_for_window(date, window_size = 36)
    df_lr      = loss_rates.generate_all_matrices()
    requires, at minimum, columns:
                           [constants.LOANID,
                            constants.LOANAMOUNT,
                            constants.GRADE,
                            constants.TERM,
                            constants.CHARGEOFF,
                            constants.AGE,
                            constants.MONTH,
                            constants.CUMUL_PRIN,
                            constants.DEFAULTMONTH,
                            constants.COMPLETEDMONTH,
                            constants.CHARGEOFFMONTH,
                            constants.ORIGMID,
                            constants.STATUS
                            ]
    """

    def __init__(self, df, run_type):
        self.run_type = run_type
        self.df = df
        self.clean_data()

    def clean_data(self):

        # Perform data selection
        selection = ((self.df[constants.TERM].isin(constants.selected_terms)) & (self.df[constants.AGE] == self.df[
            constants.TERM]))  # Ensure we only look at loans at end of term

        self.df = self.df[selection]

        selection = (~self.df[constants.DEFAULTMONTH].isnull() & (
                    self.df[constants.DEFAULTMONTH] < self.df[constants.MONTH]))
        self.df = self.df[~selection]

        selection = (~self.df[constants.CHARGEOFFMONTH].isnull() & (
                    self.df[constants.CHARGEOFFMONTH] < self.df[constants.MONTH]))
        self.df = self.df[~selection]

        if 'ObDate' not in self.df.columns:
            self.df['ObDate'] = pd.to_datetime(self.df[constants.MONTH], format='%Y%m')

        self.df['StatusAdjusted'] = self.df.apply(lambda x: self.get_status(x['DaysPastDue'], x['ObservationMonth'],
                                                                            x['ChargeoffMonth'], x['DefaultMonth']),
                                                  axis=1)

        # ADDING CODE TO OVERRIDE THE STATUS HERE SEE CODE ABOVE
        self.overwrite_status()

        if constants.LOSS_RATE_MAT not in self.df.columns:
            self.df[constants.LOSS_RATE_MAT] = self.df[constants.CHARGEOFF] / (
                        self.df[constants.LOANAMOUNT] - self.df[constants.CUMUL_PRIN])
            self.df[constants.LOSS_RATE_MAT] = self.df.apply(
                lambda x: self.fix_loss_rate(x[constants.LOSS_RATE_MAT], x[constants.LOANAMOUNT],
                                             x[constants.CUMUL_PRIN]), axis=1)

    def generate_matrices_for_window(self, window_end, window_size=6):

        # Perform time window selection
        window_start = window_end - relativedelta(months=window_size)
        df_selection = self.df[(self.df['ObDate'] > window_start) & (self.df['ObDate'] <= window_end)]

        if self.run_type != constants.LC_NEAR_PRIME_36:
            min_orig_date = window_end - relativedelta(years=6)
            selection = (df_selection[constants.ORIGMID] >= int(min_orig_date.strftime('%Y%m')))
            initial_size = df_selection.shape
            df_selection = df_selection[selection]
            end_size = df_selection.shape
            # print('Size went from %s to %s' % (str(initial_size), str(end_size)))
            if initial_size != end_size:
                print('Size has changed')

        if self.run_type in [constants.LC_PRIME_36, constants.LC_NEAR_PRIME_36]:
            # The lending club data can be full of nulls which should be filtered out before calculating
            # the loss rate. This is due to the differences between lending club and prosper data
            df_selection.loc[pd.isnull(df_selection[constants.LOSS_RATE_MAT]), constants.LOSS_RATE_MAT] = 0

        grouped_df = df_selection[[constants.TERM, constants.GRADE, 'StatusAdjusted', constants.LOSS_RATE_MAT]] \
            .groupby([constants.TERM, constants.GRADE, 'StatusAdjusted'])

        loss_rate_matrices = grouped_df.mean().clip(0, None)
        loss_rate_matrices[constants.MEASURED_DATE] = int(pd.to_datetime(window_end).to_pydatetime().strftime('%Y%m'))
        return loss_rate_matrices

    # def generate_matrices_for_window(self, window_end, window_size=6):
    #
    #     # Perform time window selection
    #     window_start = window_end - relativedelta(months=window_size)
    #     df_selection = self.df[(self.df['ObDate'] > window_start) & (self.df['ObDate'] <= window_end)]
    #
    #     min_orig_date = window_end - relativedelta(years=6)
    #     selection = (df_selection[constants.ORIGMID] >= int(min_orig_date.strftime('%Y%m')))
    #     initial_size = df_selection.shape
    #     df_selection = df_selection[selection]
    #     end_size = df_selection.shape
    #     if initial_size != end_size:
    #         print('Size has changed')
    #
    #     total_deals = initial_size[0]
    #     deals_removed = initial_size[0] - end_size[0]
    #     deals_left = end_size[0]
    #
    #     result = {
    #         'Date': window_end,
    #         'Total_deals': total_deals,
    #         'Deals_removed': deals_removed,
    #         'Deals_remaining': deals_left,
    #     }
    #     return result

    def generate_all_matrices(self, window_size=6):  # window_size in months

        self.clean_data()
        df_full = None

        for date in self.df.ObDate.unique():
            print(date)
            if date == np.datetime64('NaT'):
                continue

            window_end = pd.to_datetime(date).to_pydatetime()

            df = self.generate_matrices_for_window(window_end, window_size)
            if df_full is None:
                df_full = df
            else:
                df_full = df_full.append(df)
        return df_full

        # def generate_all_matrices(self, window_size=6): # window_size in months
        #
        #     self.clean_data()
        #     df_full = None
        #
        #     results = []
        #     for date in self.df.ObDate.unique():
        #         print(date)
        #         if date == np.datetime64('NaT'):
        #             continue
        #
        #         window_end = pd.to_datetime(date).to_pydatetime()
        #
        #         # df = self.generate_matrices_for_window(window_end, window_size)
        #         # if df_full is None:
        #         #     df_full = df
        #         # else:
        #         #     df_full = df_full.append(df)
        #
        #         result = self.generate_matrices_for_window(window_end, window_size)
        #         results.append(result)
        #     df_full = pd.DataFrame(results)
        #     pd.to_csv(df_full, 'D:\\AllData_July\\Validation_August_2019\\Data\\DealCounts.csv')

        return df_full

    @staticmethod
    def get_status(dpd, month, chargeoffmonth, defaultmonth):
        if month in {chargeoffmonth, defaultmonth}:
            return constants.STATUS_DEFAULT
        if dpd == 0:
            return constants.STATUS_CURRENT
        if dpd <= 15:
            return constants.STATUS_GRACE
        elif 16 <= dpd <= 30:
            return constants.STATUS_LATE16_30
        elif 31 <= dpd <= 120:
            return constants.STATUS_LATE31_120
        elif dpd > 120:
            return constants.STATUS_DEFAULT

    @staticmethod
    def fix_loss_rate(loss_rate, loan_amount, cumul_prin):
        if loan_amount <= cumul_prin:
            return 0
        else:
            return loss_rate

    def overwrite_status(self):
        '''
        Function to Override Status Field of the loans e.g. For LC Prime 36 Loans Grace Loans and Late30 should be
        valued together and the Grace Loans haircut is just 50% of Late30 loans, therefore override Grace loans to be
        Late30 loans
        :return:
        '''
        if self.run_type in constants.GRACE_TO_LATE30_OVERRIDE:
            self.overwrite_column('StatusAdjusted', constants.STATUS_GRACE, constants.STATUS_LATE16_30)
        return self.df

    def overwrite_column(self, column_name, current_value, new_value):
        '''
        Override values in a column to a new value. Function used for overrides
        :param column_name: Name of the column we want to override
        :param current_value: Value that will be overridden
        :param new_value: New value to override
        '''
        status_index = self.df.columns.get_loc(column_name)
        new_status = self.df.iloc[:, status_index].replace(current_value, new_value)
        self.df.iloc[:, status_index] = new_status


class SimplifiedMethodologyLossRate(LossRateMatrices):
    def __init__(self, df, run_type):
        if run_type != constants.LC_NEAR_PRIME_60:
            raise Exception('Simplified methodology should only be used Near Prime 60 loans')
        self.run_type = run_type
        self.df = df
        self.clean_data()

    def clean_data(self):
        # simplified methodolgy only related to 60 month loans
        self.df = self.df[self.df[constants.TERM] == 60]  # TODO HACK
        # self.df = self.df[self.df[constants.TERM] == 36] # TODO HACK

        if 'ObDate' not in self.df.columns:
            self.df['ObDate'] = pd.to_datetime(self.df[constants.MONTH], format='%Y%m')

    def generate_loss_rate(self, window_end, observation_window=3, look_back_window=9):

        df = self.df
        df_sample = df[(df.MONTH.isin(
            df[["MONTH"]].drop_duplicates(keep='last').sort_values("MONTH").tail(13).head(3).MONTH)) &
                       (df[constants.TERM] == 60) &
                       # (df[constants.TERM] == 36) &
                       (df.PERIOD_END_LSTAT != "Fully Paid") &
                       (df.PERIOD_END_LSTAT != "Issued") &
                       (df.PERIOD_END_LSTAT != "Charged Off") &
                       (df.PERIOD_END_LSTAT != "Default")
                       ].copy()

        df_sample.loc[df_sample.PERIOD_END_LSTAT == "In Grace Period", "PERIOD_END_LSTAT"] = "Late (16-30 days)"

        # drop COAMT column, as the proper COAMT will be joined later
        df_sample = df_sample.drop('COAMT', 1)

        # Observation month, 9 months on from actual observation date
        df_sample["MONTH_OBS_temp"] = df_sample.ObDate.dt.year * 12 + df_sample.ObDate.dt.month + 9
        df_sample["temp_yr"] = np.floor(df_sample.MONTH_OBS_temp / 12)
        df_sample["temp_month"] = df_sample.MONTH_OBS_temp - np.floor(df_sample.MONTH_OBS_temp / 12) * 12 + 1
        df_sample["MONTH_OBS"] = pd.to_datetime(
            df_sample['temp_yr'].astype(int).astype(str) + df_sample['temp_month'].astype(int).astype(str),
            format='%Y%m')

        # gather all charged-off amounts observed for the loans in the sample
        df_coamt = df[(df[constants.LOANID].isin(df_sample[constants.LOANID])) & (df.COAMT != 0)].copy()
        df_coamt = df_coamt[[constants.LOANID, "ObDate", "COAMT"]]
        df_coamt.rename(index=str, columns={"ObDate": "MONTH_CO"}, inplace=True)
        df_coamt.set_index(constants.LOANID, inplace=True)

        df_sample = df_sample.join(df_coamt, on=[constants.LOANID])
        df_sample.loc[
            (df_sample.COAMT != 0) &
            ((df_sample.MONTH_CO <= df_sample.ObDate) | (df_sample.MONTH_CO > df_sample.MONTH_OBS)),
            "COAMT"] = 0

        # IF COAMT is NaN, replace with zero
        df_sample.loc[pd.isnull(df_sample.COAMT), "COAMT"] = 0

        # total charged-off amount observed in the 9-month period is divided by the principal balance at the
        # beginning of the period
        loss_rates = df_sample.groupby([constants.GRADE, "PERIOD_END_LSTAT"]).sum().reset_index()
        loss_rates["loss_rate_before_recovery"] = loss_rates[constants.CHARGEOFF] / loss_rates["PBAL_END_PERIOD"]
        return loss_rates

    def generate_backtesting_loss_rates(self):
        df_full = None

        for date in sorted(self.df.ObDate.unique(), reverse=True):

            print(date)
            if date == np.datetime64('NaT'):
                continue

            window_end = pd.to_datetime(date).to_pydatetime()

            df = self.generate_loss_rate(window_end)

            if df_full is None:
                df_full = df
            else:
                df_full = df_full.append(df)

        return df_full


class Biz2CreditLossRate(SimplifiedMethodologyLossRate):
    def __init__(self, df, run_type, run_with_prosper=False):
        """
        Initialise the Class
        :param df: Combined LoanTape and Payments File
        :param run_type: Which loans are we running
        :param run_with_prosper: Are we running Prosper Loans via an alternative Methodology
        """
        if run_type != constants.BIZ2CREDIT:
            raise Exception('Biz2Credit simplified methodology should only be used for Biz2Credit loans')
        self.run_type = run_type
        self.df = df
        self.run_with_prosper = run_with_prosper
        self.clean_data()

    def clean_data(self):
        """
        Function to reformat the dates
        :return:
        """
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'], format='%Y-%m-%d')

    def calculate_for_one_window(self, start_date, end_date):
        '''
        Function to generate table of all results that should be grouped by for loss rate calculate
        :param start_date: start date of the window
        :param end_date: end date of the window
        :return: pandas dataframe of loans containing their term group, writeoff amount and principle outstanding
        '''
        # Remove payments known as of start_date
        new_df = self.df[self.df.transaction_date <= start_date].copy()
        # Get the last payment of a specific Loan and Case
        # new_df['case_id'] = new_df['case_id_x']  hack - not best solution
        new_df.sort_values(
            [constants.LOANID, 'case_id', 'transaction_date', 'early_payoff', 'description'],
            ascending=[True, True, False, False, False], inplace=True
        )
        new_df = new_df.groupby([constants.LOANID, 'case_id']).first()
        # Keep only loans that are: i) not an early payoff and ii) There is principal outstanding
        new_df = new_df[(new_df.early_payoff != 1) & (new_df.principal_outstanding != 0)]
        # Remove loans whose write off date is less than start date
        new_df = new_df[~((pd.to_datetime(new_df.write_off_date) <= start_date) & (~pd.isnull(new_df.write_off_date)))]

        if not self.run_with_prosper and not new_df.empty:
            new_df['date_diff'] = (start_date - new_df['transaction_date']).dt.days

            new_df['use_method_a'] = new_df['transaction_date'].dt.month == start_date.month
            # new_df['frequency'] = new_df['frequency_x']  hack - not best solution
            new_df['adjusted_dpd'] = new_df.apply(lambda x: self.adjusted_days_past_due(
                x['use_method_a'],
                x['dpd'],
                x['date_diff'],
                x['frequency']  # removed '_x'
            ), axis=1) # iterating over rows so axis=1

            new_df[constants.STATUS] = new_df.apply(lambda x: self.get_status(
                x['adjusted_dpd'],
            ), axis=1)

        new_df['write_off_amount'] = new_df['gross_write_off']
        new_df.loc[pd.isnull(new_df['write_off_date']), 'write_off_amount'] = 0
        new_df.loc[pd.to_datetime(new_df['write_off_date']) > end_date, 'write_off_amount'] = 0
        results = new_df

        return results

    def calculate_loss_rate(self, start_date, time_period=9, number_of_windows=6):
        '''
        :param start_date: start date to calculate for
        :param time_period: Length between date1 and date 2 in months
        :param number_of_windows: Number of windows to include in total calculation
        :return: Dataframe of loss rates without recovery for each expected term and loan status
        '''
        dates_dictionary = {
            start_date + pd.tseries.offsets.MonthEnd(-(i + time_period)):
                start_date + pd.tseries.offsets.MonthEnd(-i) for i in
            range(number_of_windows)}
        final_table = None
        for sd, ed in dates_dictionary.items():
            table = self.calculate_for_one_window(sd, ed)
            if isinstance(final_table, pd.DataFrame):
                final_table = final_table.append(table)
            else:
                final_table = table.copy()
        grouped_results = final_table.groupby(['term_group', constants.STATUS]).sum()
        writeoff = grouped_results.write_off_amount
        principal_outstanding = grouped_results.principal_outstanding
        final_results = (writeoff / principal_outstanding).reset_index(name="loss_rate_without_recovery")
        return final_results

    def generate_backtesting_loss_rates(self, dates_to_run):
        date_mapping = {convert_date(d): d for d in dates_to_run}
        all_dates = date_mapping.keys()
        results_dictionary = {}
        for date in all_dates:
            result = self.calculate_loss_rate(date)
            date_int = date_mapping.get(date)
            results_dictionary[date_int] = result
        return results_dictionary

    @staticmethod
    def adjusted_days_past_due(use_method_a, dpd, diff_days, frequency):
        adjusted_dpd = dpd + diff_days
        if not dpd:
            if use_method_a:
                adjusted_dpd = 0
            else:
                adjusted_dpd += constants.ADJUSTED_DPD[frequency]
        return max(adjusted_dpd, 0)

    @staticmethod
    def get_status(dpd):
        status = constants.BIZ2CREDIT_STATUS_LATE61_PLUS
        if dpd == 0:
            status = constants.BIZ2CREDIT_STATUS_CURRENT
        elif dpd <= 15:
            status = constants.BIZ2CREDIT_STATUS_GRACE
        elif 16 <= dpd <= 30:
            status = constants.BIZ2CREDIT_STATUS_LATE16_30
        elif 31 <= dpd <= 60:
            status = constants.BIZ2CREDIT_STATUS_LATE30_60
        return status

    @staticmethod
    def get_writeoff_amount(write_off_date, write_off_amount, end_date):
        use_write_off = not pd.isnull(write_off_date) and write_off_date <= end_date
        return write_off_amount if use_write_off else 0


class IOULossRate(Biz2CreditLossRate):
    def __init__(self, df, run_type, run_with_prosper=False):
        if run_type != constants.IOU_LOANS:
            raise Exception('IOU simplified methodology should only be used for IOU loans')
        self.run_type = run_type
        self.df = df
        self.run_with_prosper = run_with_prosper
        self.clean_data()

    def clean_data(self):
        self.df['write_off_date'] = pd.to_datetime(self.df['write_off_date'], format='%Y-%m-%d')
        self.df['month_end_date'] = pd.to_datetime(self.df['month_end_date'], format='%Y-%m-%d')

    def calculate_for_one_window(self, start_date, end_date):
        '''
        Function to generate table of all results that should be grouped by for loss rate calculate
        :param start_date: start date of the window
        :param end_date: end date of the window
        :return: results pandas dataframe of loans containing their term group, writeoff amount and principle outstanding
        '''

        new_df = self.df.copy(True)
        new_df = new_df[new_df.month_end_date == start_date]
        new_df = new_df[~((new_df.write_off_date <= start_date) & (~pd.isnull(new_df.write_off_date)))]

        new_df = new_df[new_df.start_amount > new_df.total_principal_paid]

        new_df['principal_outstanding'] = new_df.start_amount - new_df.total_principal_paid

        new_df['write_off_amount'] = new_df['gross_write_off']
        new_df.loc[pd.isnull(new_df['write_off_date']), 'write_off_amount'] = 0
        new_df.loc[new_df['write_off_date'] > end_date, 'write_off_amount'] = 0
        results = new_df

        return results