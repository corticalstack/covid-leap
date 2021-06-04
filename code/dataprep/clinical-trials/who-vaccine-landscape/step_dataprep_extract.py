import numpy as np
import pandas as pd
import camelot
import argparse
import os
import hashlib
from azureml.core import Run
from azure.storage.blob import BlobServiceClient
import copy
import re
import glob
import json
from datetime import datetime


class Load:
    def __init__(self):
        self.run = Run.get_context()
        self.args = None
        self.get_runtime_arguments()

        self.all_who = None
        self.pdf_extract = None
        self.xlsx_extract_clinical = None
        self.xlsx_extract_preclinical = None

        self.articles_in = 0
        self.articles_out = 0

        self.clinical_in = 0
        self.pre_clinical_in = 0
        self.total_clinical_in = 0
        self.total_pre_clinical_in = 0

        self.evaluations = []

        self.eval_type_clinical = 'clinical'
        self.clinical_table_type = None
        self.clinical_deconstruct_complete = None

        self.eval_type_pre_clinical = 'pre-clinical'
        self.pre_clinical_table_type = None
        self.pre_clinical_deconstruct_complete = None

        self.candidate = {'id': None,
                          'evaluation_type': None,
                          'metadata': {
                            'disease': self.args.disease,
                            'manufacturer': None,
                            'platform': None,
                            'type': None,
                            'number_doses': None,
                            'timing_doses': None,
                            'route_administration': None,
                            'shared_platform': None,
                            'phase': '',
                            'clinical_stage': []
                          }
                          }

        self.trial = {'phase': None,
                      'id': None
                      }

        self.blob_connect_string = None
        self.get_os_environ_variables()

        self.process()

    @staticmethod
    def strip_line_break(str_in):
        str_out = str_in.replace('\n', '')
        return str_out

    @staticmethod
    def strip_characters(str_in):
        str_out = str_in.replace('\n', '')
        str_out = str_out.replace('\t', '')
        str_out = str_out.replace('\"', '')
        str_out = str_out.replace('\xa0', '')
        str_out = str_out.replace('\\', '')
        str_out = str_out.replace('*', '')
        str_out = str_out.lstrip()
        str_out = str_out.rstrip()
        return str_out

    @staticmethod
    def strip_unicode_characters(str_in):
        str_out = str_in.encode('ascii', 'ignore')
        str_out = str_out.decode()
        return str_out

    @staticmethod
    def strip_multiple_space(str_in):
        str_out = re.sub(' +', ' ', str_in)
        return str_out

    @staticmethod
    def strip_dash(str_in):
        str_out = str_in.replace('\u2013', '')
        return str_out

    @staticmethod
    def strip_other(str_in):
        str_out = re.sub('[çãı]', '', str_in)
        str_out = str_out.lstrip()
        str_out = str_out.rstrip()
        return str_out

    @staticmethod
    def get_hash_id(et, key_string):
        encoded = str(et + key_string).encode('utf-8')
        return hashlib.sha1(encoded).hexdigest()

    def get_runtime_arguments(self):
        print('--- Get Runtime Arguments')
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--input',
            type=str,
            help='Path to input data'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Path to output data'  # if output is file based rather than blob
        )
        parser.add_argument(
            '--output_dataset',
            type=str,
            help='Path to output dataset'
        )
        parser.add_argument(
            '--disease',
            type=str,
            help='Disease landscape identifier'
        )

        self.args = parser.parse_args()

        print('Input: {}'.format(self.args.input))
        print('Output Dataset: {}'.format(self.args.output_dataset))
        print('Disease: {}'.format(self.args.disease))

    def get_os_environ_variables(self):
        self.blob_connect_string = os.environ.get('AZ_RP_MLW_BLOB_CONNECT_STRING')

    def process(self):
        self.get_who_landscape_article_list()
        self.collect_metrics_pre()

        for idx, entry in enumerate(self.all_who):
            print('\nExtracting raw WHO landscape article {}'.format(entry))
            self.reset_article_evaluation()

            if '.pdf' in entry:
                self.get_tables_from_pdf(entry)
                self.pdf_tables_deconstruct()
            elif '.xlsx' in entry:
                self.get_tables_from_xlsx(entry)
                self.xlsx_deconstruct()

            #self.output_eval_as_json(entry)
            self.output_eval_as_json_blob(entry)
            self.log_metrics_per_article_post()

        self.log_metrics_post()

    def get_who_landscape_article_list(self):
        print('--- Get WHO Landscape Article List')
        self.all_who = glob.glob(f'{self.args.input}/novel-coronavirus-landscape-covid-19-2021*.*', recursive=True)
        print('Number WHO Vaccine Landscape Articles {}'.format(str(len(self.all_who))))

    def reset_article_evaluation(self):
        self.clinical_deconstruct_complete = None
        self.pre_clinical_deconstruct_complete = None
        self.evaluations = []
        self.clinical_in = 0
        self.pre_clinical_in = 0

    def get_tables_from_xlsx(self, entry):
        self.xlsx_extract_clinical = pd.read_excel(entry, 'Clinical', engine='openpyxl')
        self.xlsx_extract_preclinical = pd.read_excel(entry, 'Pre-Clinical', engine='openpyxl')

    def xlsx_deconstruct(self):
        self.xlsx_deconstruct_landscape_clinical()
        self.xlsx_deconstruct_landscape_preclinical()

    def xlsx_deconstruct_landscape_clinical(self):
        et = self.eval_type_clinical
        phase_col_upperbound = 14
        trial_phase = {9: '1', 10: '1/2', 11: '2', 12: '2/3', 13: '3', 14: '4'}
        phase_map = {'Phase 1': '1', 'Phase 1/2': '1/2', 'Phase 2': '2', 'Phase 2/3': '2/3', 'Phase 2b/3': '2b/3',
                     'Phase 3': '3', 'Phase 4': '4'}

        omit = ['Study Report', 'Study Report1', 'Study Report2', 'Interim Report', 'Final Report',
                '(phase 2b)', 'Phase2b/3', '(not yet recruiting)', 'Report', 'Development', 'Pre-clinical result']

        last_trial = pd.Series(dtype=str)

        try:
            self.xlsx_extract_clinical.columns = ['id', 'platform_acronym', 'platform', 'type', 'number_doses',
                                                  'timing_doses', 'route_administration', 'manufacturer', 'phase', '1', '1/2',
                                                  '2', '2/3', '3', 'confirmed_symptomatic_cases',
                                                  'efficacy_compared_placebo', 'efficacy_severe_admissions',
                                                  'efficacy_prevention_emergency_visit', 'efficacy_covid19',
                                                  'efficacy_deaths', 'efficacy_seronversion', 'humoral_immunogenicity',
                                                  'safety_booster_dose', 'dummy']
        except ValueError as e:
            try: # Note phase 4 introduced in file 20210226
                self.xlsx_extract_clinical.columns = ['id', 'platform_acronym', 'platform', 'type', 'number_doses',
                                                      'timing_doses', 'route_administration', 'manufacturer', 'phase',
                                                      '1', '1/2', '2', '2/3', '3', '4', 'confirmed_symptomatic_cases',
                                                      'efficacy_compared_placebo', 'efficacy_severe_admissions',
                                                      'efficacy_prevention_emergency_visit', 'efficacy_covid19',
                                                      'efficacy_deaths', 'efficacy_seronversion',
                                                      'humoral_immunogenicity', 'safety_booster_dose', 'dummy']
                phase_col_upperbound = 15
            except ValueError as e:
                try:  # Note phase not reported introduced in file 20210406
                    self.xlsx_extract_clinical.columns = ['id', 'platform_acronym', 'platform', 'type', 'number_doses',
                                                          'timing_doses', 'route_administration', 'manufacturer',
                                                          'phase',
                                                          '1', '1/2', '2', '2/3', '3', '4', 'ph_not_reported',
                                                          'confirmed_symptomatic_cases',
                                                          'efficacy_compared_placebo', 'efficacy_severe_admissions',
                                                          'efficacy_prevention_emergency_visit', 'efficacy_covid19',
                                                          'efficacy_deaths', 'efficacy_seronversion',
                                                          'humoral_immunogenicity', 'safety_booster_dose', 'dummy']
                    phase_col_upperbound = 15
                except ValueError as e:
                    print('Fatal error defining xlsx columns: {}'.format(e))
                    return

        new_candidate = {}
        for index, row in self.xlsx_extract_clinical.iterrows():
            if isinstance(row['id'], int):
                try:
                    if len(new_candidate['metadata']['clinical_stage']) > 0:
                        self.evaluations.append(new_candidate)
                        self.clinical_in += 1
                except (UnboundLocalError, KeyError) as e:
                    pass

                row['manufacturer'] = self.strip_characters(row['manufacturer'])
                row['manufacturer'] = self.strip_unicode_characters(row['manufacturer'])

                row['type'] = self.strip_characters(row['type'])
                row['type'] = self.strip_unicode_characters(row['type'])

                last_trial = row[1:8]
                new_candidate = copy.deepcopy(self.candidate)
                new_candidate['id'] = self.get_hash_id(et, row['manufacturer'] + row['platform'] + row['type'])
                new_candidate['evaluation_type'] = et
                new_candidate['metadata']['manufacturer'] = row['manufacturer']
                new_candidate['metadata']['platform'] = row['platform']
                new_candidate['metadata']['type'] = row['type']

                if type(row['number_doses']) is int or type(row['number_doses']) is float or \
                        type(row['number_doses']) is str:
                    new_candidate['metadata']['number_doses'] = row['number_doses']
                else:
                    new_candidate['metadata']['number_doses'] = 'ND'

                new_candidate['metadata']['timing_doses'] = row['timing_doses']
                new_candidate['metadata']['route_administration'] = row['route_administration']
                new_candidate['metadata']['shared_platform'] = None

                row['phase'] = self.strip_characters(row['phase'])
                new_candidate['metadata']['phase'] = phase_map[row['phase']]

            if not last_trial.empty:
                for ph in range(9, phase_col_upperbound):
                    if pd.isnull(row[ph]):
                        continue
                    temp_id = row[ph]
                    temp_id = self.strip_characters(temp_id)
                    if temp_id in omit:
                        continue
                    temp_id = temp_id.split(';')  # Rare occurrences of multiple trials ids same line sep by ;
                    for idx, i in enumerate(temp_id):
                        new_trial = copy.deepcopy(self.trial)
                        new_trial['phase'] = trial_phase[ph]
                        new_trial['id'] = temp_id[idx]
                        new_candidate['metadata']['clinical_stage'].append(new_trial)

        try:
            if len(new_candidate['metadata']['clinical_stage']) > 0:
                self.evaluations.append(new_candidate)
                self.clinical_in += 1
        except UnboundLocalError:
            pass

    def xlsx_deconstruct_landscape_preclinical(self):
        et = self.eval_type_pre_clinical
        self.xlsx_extract_preclinical.columns = ['id', 'platform_acronym', 'platform', 'type', 'target',
                                                 'shared_platform', 'manufacturer', 'd8', 'd9', 'd10', 'd11', 'd12',
                                                 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20', 'd21', 'd22',
                                                 'd23', 'd24']

        self.xlsx_extract_preclinical = self.xlsx_extract_preclinical.replace(np.nan, '')
        for index, row in self.xlsx_extract_preclinical.iterrows():
            if not isinstance(row['id'], int):
                continue
            if row['type'] is '':
                row['type'] = 'TBD'

            row['manufacturer'] = self.strip_characters(row['manufacturer'])
            row['manufacturer'] = self.strip_unicode_characters(row['manufacturer'])
            row['manufacturer'] = self.strip_other(row['manufacturer'])

            row['type'] = self.strip_characters(row['type'])
            row['type'] = self.strip_unicode_characters(row['type'])
            row['type'] = self.strip_other(row['type'])

            new_candidate = copy.deepcopy(self.candidate)
            new_candidate['id'] = self.get_hash_id(et, row['manufacturer'] + row['platform'] + row['type'])
            new_candidate['evaluation_type'] = et
            new_candidate['metadata']['manufacturer'] = row['manufacturer']
            new_candidate['metadata']['platform'] = row['platform']
            new_candidate['metadata']['type'] = row['type']
            new_candidate['metadata']['number_doses'] = None
            new_candidate['metadata']['timing_doses'] = None
            new_candidate['metadata']['route_administration'] = None
            if row['shared_platform']:
                new_candidate['metadata']['shared_platform'] = row['shared_platform']

            self.evaluations.append(new_candidate)
            self.pre_clinical_in += 1

    def get_tables_from_pdf(self, entry):
        self.pdf_extract = camelot.read_pdf(entry, pages='1-end')
        print('Number of pages in PDF: ', str(len(self.pdf_extract)))

    def pdf_add_new_candidate(self, et, dev, vpl, tcv, nd, td, ra, sp):
        new_candidate = copy.deepcopy(self.candidate)

        if dev:
            dev = self.strip_line_break(dev)
            dev = self.strip_dash(dev)
            dev = self.strip_other(dev)
            dev = self.strip_multiple_space(dev)
        if vpl:
            vpl = self.strip_line_break(vpl)
        if tcv:
            tcv = self.strip_line_break(tcv)
            tcv = self.strip_multiple_space(tcv)
        if nd:
            nd = self.strip_line_break(nd)
        if td:
            td = self.strip_line_break(td)
        if ra:
            ra = self.strip_line_break(ra)
        if sp:
            sp = self.strip_line_break(sp)

        if vpl == '':
            if 'Protein Subunit' in dev:
                dev = dev.replace('Protein Subunit', '')
                vpl = 'Protein Subunit'

        new_candidate['id'] = self.get_hash_id(et, dev + vpl + tcv)

        new_candidate['evaluation_type'] = et
        new_candidate['metadata']['manufacturer'] = dev
        new_candidate['metadata']['platform'] = vpl
        new_candidate['metadata']['type'] = tcv
        new_candidate['metadata']['number_doses'] = nd
        new_candidate['metadata']['timing_doses'] = td
        new_candidate['metadata']['route_administration'] = ra
        new_candidate['metadata']['shared_platform'] = sp

        self.evaluations.append(new_candidate)

        if et == self.eval_type_clinical:
            self.clinical_in += 1
        elif et == self.eval_type_pre_clinical:
            self.pre_clinical_in += 1

        return new_candidate

    def add_new_trial_type0(self, ca, tr):
        trials_temp = tr.split('\n')
        phases = ['Phase 1', 'Phase 1/2', 'Phase 2', 'Phase 3']
        phase_map = {'Phase 1': '1', 'Phase 1/2': '1/2', 'Phase 2': '2', 'Phase 2b/3': '2b/3', 'Phase 3': '3'}
        current_phase = ''
        for t in trials_temp:
            if any(x in t for x in phases):
                current_phase = t.rstrip()
                continue
            new_trial = copy.deepcopy(self.trial)
            try:
                new_trial['phase'] = phase_map[current_phase]
            except KeyError:
                continue
            new_trial['id'] = t
            ca['metadata']['clinical_stage'].append(new_trial)
            if new_trial['phase'] > ca['metadata']['phase']:
                ca['metadata']['phase'] = new_trial['phase']

    def add_new_trial_type1(self, ca, ph, tr):
        trial_phase = {6: '1', 7: '1/2', 8: '2', 9: '3'}
        omit = ['Study Report', 'Study Report1', 'Study Report2', 'Interim Report', 'Final Report', '(phase 2b)',
                'Phase2b/3', '(not yet recruiting)']

        # Handle PDF transformation odd cases
        if 'N\nCT' in tr:
            tr = tr.replace('\n', '')

        trials = tr.split('\n')
        trials = [x.strip(' ') for x in trials]
        trials = [t for t in trials if t not in omit]

        for t in trials:
            new_trial = copy.deepcopy(self.trial)
            new_trial['phase'] = trial_phase[ph]
            t = self.strip_characters(t)
            new_trial['id'] = t
            ca['metadata']['clinical_stage'].append(new_trial)
            if new_trial['phase'] > ca['metadata']['phase']:
                ca['metadata']['phase'] = new_trial['phase']

    def deconstruct_landscape(self, table):
        last_trial = pd.Series(dtype=str)
        new_candidate = None

        for index, row in table.iterrows():
            # Start of table info for vaccines in clinical trials
            if row[0] == 'COVID-19 Vaccine \ndeveloper/manufacturer' and 'Vaccine platform' in row[1]:
                self.clinical_table_type = 1
                self.clinical_deconstruct_complete = False
                continue

            if row[0] == 'Platform' and row[1] == 'Type of \ncandidate \nvaccine':
                self.clinical_table_type = 0
                self.clinical_deconstruct_complete = False
                continue

            # Start of table info for pre-clinical vaccines
            if row[0] == 'Platform' and row[1] == 'Type of candidate vaccine':
                self.clinical_deconstruct_complete = True
                self.pre_clinical_table_type = 1
                self.pre_clinical_deconstruct_complete = False
                continue

            if row[0] == 'Platform' and row[1] == 'Type of candidate \nvaccine':
                self.clinical_deconstruct_complete = True
                self.pre_clinical_table_type = 0
                self.pre_clinical_deconstruct_complete = False
                continue

            if not self.clinical_deconstruct_complete:
                # Blank header row
                if row[0] == '' and last_trial.empty:
                    continue

                # Ingest clinical trial
                if row[0] != '' and not row[0:3].equals(last_trial):
                    if self.clinical_table_type == 0:
                        dev = row[2]
                        vpl = row[0]
                        tcv = row[1]
                        nd = None
                        td = None
                        ra = None
                        sp = None
                        if row[5] != '':
                            sp = row[5]
                    else:
                        dev = row[0]
                        vpl = row[1]
                        tcv = row[2]
                        nd = row[3]
                        td = row[4]
                        ra = row[5]
                        sp = None
                    last_trial = row[0:3]

                    # Handle empty row on table that spans pages
                    if not [x for x in (dev, tcv) if x == '']:
                        new_candidate = self.pdf_add_new_candidate('clinical', dev, vpl, tcv, nd, td, ra, sp)
                    else:
                        continue

                if self.clinical_table_type == 0:
                    self.add_new_trial_type0(new_candidate, row[4])
                else:
                    for i in range(6, 10):
                        if row[i]:
                            self.add_new_trial_type1(new_candidate, i, row[i])

            # Ingest pre-clinical
            if self.clinical_deconstruct_complete and not self.pre_clinical_deconstruct_complete:
                dev = row[2]
                vpl = row[0]
                tcv = row[1]
                sp = None
                if row[5] != '':
                    sp = row[5]
                if not [x for x in (dev, tcv) if x == '']:
                    self.pdf_add_new_candidate('pre-clinical', dev, vpl, tcv, None, None, None, sp)

    def pdf_tables_deconstruct(self):
        for t in self.pdf_extract:
            self.deconstruct_landscape(t.df)

    def collect_metrics_pre(self):
        self.articles_in = len(self.all_who)

    def log_metrics_per_article_post(self):
        print('Number clinical out', self.clinical_in)
        print('Number pre-clinical out', self.pre_clinical_in)

        self.total_clinical_in += self.clinical_in
        self.total_pre_clinical_in += self.pre_clinical_in
        self.articles_out += 1

    def log_metrics_post(self):
        self.run.log('# WHO articles evaluated', self.articles_in)
        self.run.log('# Clinical evaluations', self.total_clinical_in)
        self.run.log('# Pre-clinical evaluations', self.total_pre_clinical_in)
        self.run.log('# Trials articles created', self.articles_out)

    def output_eval_as_json(self, entry):
        json_file = entry
        json_file = json_file.replace('.pdf', '.json')
        json_file = json_file.replace('.xlsx', '.json')
        json_file = json_file.replace(self.args.input, self.args.output)
        print('Outputting JSON file {}'.format(json_file))
        with open(json_file, 'w', encoding='utf-8', newline='\r\n') as outfile:
            json.dump(self.evaluations, outfile, indent=4)

    def output_eval_as_json_blob(self, entry):
        json_file = entry
        json_file = json_file.replace('.pdf', '.json')
        json_file = json_file.replace('.xlsx', '.json')
        json_file = json_file.replace(self.args.input, '')
        print('Outputting JSON file {}'.format(json_file))

        container_name = 'datasets'
        blob_name = self.args.output_dataset + '/' + json_file
        blob_service_client = BlobServiceClient.from_connection_string(self.blob_connect_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        temp_json = json.dumps(self.evaluations, indent=4)
        blob_client.upload_blob(temp_json, overwrite=True)


if __name__ == '__main__':
    print('--- Load Started')
    load = Load()
    print('--- Load Completed')
