import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from lxml import etree
from collections import defaultdict
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
import json
from pm4py.objects.log.importer.xes import importer as xes_importer
import pickle
from sklearn.metrics import f1_score
from hyperopt import tpe
import numpy as np
from hyperopt import Trials, hp, fmin
from hyperopt.pyll import scope
from sklearn.model_selection import train_test_split


GOAL2NUMBER = {'Other, see explanation': 0,
         'Home improvement': 1,
         'Car': 2,
         'Existing loan takeover': 3,
         'Not speficied': 4,
         'Unknown': 5,
         'Caravan / Camper': 6,
         'Extra spending limit': 7,
         'Motorcycle': 8,
         'Boat': 9,
         'Remaining debt home': 10,
         'Business goal': 11,
         'Tax payments': 12,
         'Debt restructuring': 13}
NUMBER2GOAL = {0:'Other, see explanation',
         1:'Home improvement',
         2:'Car',
         3:'Existing loan takeover',
         4:'Not speficied',
         5:'Unknown',
         6:'Caravan / Camper',
         7:'Extra spending limit',
         8:'Motorcycle',
         9:'Boat',
         10:'Remaining debt home',
         11:'Business goal',
         12:'Tax payments',
         13: 'Debt restructuring'}

APPLICATION_TYPE2NUMBER = {'New credit': 0, 'Limit raise': 1}
NUMBER2APPLICATION_TYPE = {0:'New credit', 1:'Limit raise'}


def retrieve_aligned_trace(log, net, im, fm):
    parameters = {}
    parameters[
        alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE] = True
    aligned_traces = alignments.apply_log(log, net, im, fm, parameters=parameters)

    aligned_log = []
    for trace in aligned_traces:
        new_trace = []
        for event in trace['alignment']:
            if event[0][1] != '>>':
                new_trace.append(event[0][1])
        aligned_log.append(new_trace)

    return aligned_log


def retrive_XOR_petrinet(path_petrinet):
    tree = etree.parse(path_petrinet)
    root = tree.getroot()
    net = root.find('net')
    page = net.find('page')

    xor_petri = defaultdict(list)

    for place in page.findall('place'):
        xor_petri[place.get('id')] = []

    for arc in page.findall('arc'):
        if arc.get('source') in xor_petri:
            xor_petri[arc.get('source')].append(arc.get('target'))

    remove = [key for key in xor_petri if len(xor_petri[key]) < 2]
    for r in remove:
        del xor_petri[r]
    return xor_petri


def extract_event_from_trace(trace, activity):
    i = len(trace) - 1
    while i >= 0:
        if trace[i]['concept:name'] == activity:
            return trace[i]
        else:
            i -= 1
    return None


def extract_enriched_alignment_last(original_trace, aligned_log, attrib):
    new_trace = []
    for trace in aligned_log:
        for e in trace['alignment']:
            if e[0][1] != '>>':
                if e[1][0] != '>>':
                    event_original = extract_event_from_trace(original_trace, e[1][0])
                    event = [e[0][1], event_original['role'][-1], event_original['start_timestamp'],
                             event_original['end_timestamp'], event_original['st_wip'], event_original['st_task_wip'],
                             event_original['queue'], event_original['rp_oc']]
                else:
                    event = [e[0][1], 'invisible']
                for a in attrib:
                    event.append(original_trace[0][a])
                    #event.append(original_trace.attributes[a]) ### BPI17
                new_trace.append(event)
    return new_trace


def create_prefix(traces, transitions):
    prefix_traces = {key: [] for key in transitions}
    for trace in traces:
        prefix = []
        for e in trace:
            if e[0] in transitions:
                prefix_traces[e[0]].append(prefix.copy())
            prefix.append(e)
    return prefix_traces


def find_max_length(prefix_traces):
    maximum = []
    for key in prefix_traces:
        list_len = [len(i) for i in prefix_traces[key]]
        maximum.append(max(list_len) if len(list_len) > 0 else 0)
    return max(maximum)


def trace_filter_decision_point(aligned_log, list_transitions):
    trace_filter = []
    for trace in aligned_log:
        for element in list_transitions:
            for event in trace:
                if element == event[0]:
                    trace_filter.append(trace.copy())
    return trace_filter


def retrieve_resource_to_role(path_role):
    resource_to_role = {}
    if os.path.exists(path_role):
        with open(path_role) as file:
            data = json.load(file)
            for idx, key in enumerate(data["roles"]):
                for resource in data["roles"][key]:
                    resource_to_role[resource] = int(key[len(key) - 1])
    return resource_to_role


def latest_payload_encoding(prefix_traces, attrib, activity2n, max_lenght, type=None):
    #### prefix_traces, last_resource, last_time, amount
    #### 'role', 'start_timestamp', 'end_timestamp', 'st_wip', 'st_task_wip', 'queue', 'rp_oc'
    columns = ['e_' + str(i) for i in range(1, max_lenght + 1)]
    other_feature = ['last_resource', 'last_time', 'last_wip', 'last_task_wip', 'last_queue', 'last_rp_oc']
    columns += other_feature + [a for a in attrib] + ['label']
    traces = []
    for key in prefix_traces:
        for trace in prefix_traces[key]:
            #print(trace)
            encoding_trace = [activity2n[e[0]] for e in trace]
            if len(trace) < max_lenght:
                encoding_trace = encoding_trace + ([activity2n['PAD']] * (max_lenght - len(trace)))
            length = len(trace) - 1
            flag = True
            while length > 0 and flag:
                if trace[length][1] != 'invisible':
                    time = int(trace[length][3].timestamp())
                    if type == 'test':
                        #[self.last_role, time, resource_trace.count + dict_mean['wip'], resource_task.count+ dict_mean[trans.label], queue + dict_mean[resource.get_name()+'_queue']
                        res = trace[length][1]
                        encoding_trace += [res, time] + [trace[length][4] + dict_mean['wip'], trace[length][5],  trace[length][6] + dict_mean['queue']]
                    else:
                        encoding_trace += [trace[length][1], time] + trace[length][4:-len(attrib)] #logs without attribute # + trace[length][4:-1] log with attrib
                    flag = False
                length -= 1
            if len(encoding_trace) == max_lenght:
                encoding_trace += [0]*(len(other_feature))
            for i, a in enumerate(attrib):
                if a == 'ApplicationType':
                    encoding_trace += [APPLICATION_TYPE2NUMBER[trace[0][i + 2]]]
                elif a == 'LoanGoal':
                    encoding_trace += [GOAL2NUMBER[trace[0][i + 2]]]
                else:
                    encoding_trace += [trace[0][i + 2]]
                    #encoding_trace += [trace[0][i + 7]] #per SynLoan
            encoding_trace = encoding_trace + [key]
            traces.append(encoding_trace)

    df = pd.DataFrame(traces, columns=columns)
    return df


def del_other_traces(trace_log, caseid):
    index_to_delete = []
    for trace in trace_log:
        if trace.attributes['concept:name'] != caseid:
            index_to_delete.append(trace.attributes['concept:name'])
    del trace_log[index_to_delete[0]]


### to fix file_path ####

#NAME = 'BPI_Challenge_2012_W_Two_TS'
#path_log = "/Users/francescameneghello/Desktop/BPI_Challenge_2012_W_Two_TS_AMOUNT.xes"
#path_petrinet = "/Users/francescameneghello/Documents/GitHub/RIMS/BPI_Challenge_2012_W_Two_TS/rims/BPI_Challenge_2012_W_Two_TS.pnml"
#path_role = "/Users/francescameneghello/Documents/GitHub/RIMS/BPI_Challenge_2012_W_Two_TS/rims/BPI_Challenge_2012_W_Two_TS_dispr_meta.json"
#ATTRIB = ['amount']

NAME = 'BPI_Challenge_2017_W_Two_TS'
path_log_train = "/Users/francescameneghello/Documents/GitHub/CheckerSimulation/simulation_checker/sim_regression/Queue/log_for_decision_mining/replay_log_BPI_Challenge_2017_W_Two_TS_train_queue.xes"
#path_log_train = '/Users/francescameneghello/Documents/GitHub/CheckerSimulation/simulation_checker/sim_regression/Queue/log_for_decision_mining/replay_log_confidential_1000_train_queue.xes'
path_log_test = "/Users/francescameneghello/Documents/GitHub/CheckerSimulation/simulation_checker/sim_regression/Queue/log_for_decision_mining/replay_log_BPI_Challenge_2017_W_Two_TS_test_queue.xes"
#path_log_test = '/Users/francescameneghello/Documents/GitHub/CheckerSimulation/simulation_checker/sim_regression/Queue/log_for_decision_mining/replay_log_confidential_1000_test_queue.xes'
#path_petrinet = NAME + "/rims/" + NAME + ".pnml"
path_petrinet = "/Users/francescameneghello/Documents/GitHub/RIMS/BPI_Challenge_2017_W_Two_TS/rims_plus/BPI_Challenge_2017_W_Two_TS.pnml"
path_role = NAME + "/rims/" + NAME + "_diapr_meta.json"
ATTRIB = ['amount', 'ApplicationType', 'LoanGoal']
#ATTRIB = ['amount']


##### training log #####
train_log = xes_importer.apply(path_log_train)
print('TRAIN LOG', len(train_log))
test_log = xes_importer.apply(path_log_test)
print('TEST LOG', len(test_log))
net, im, fm = pm4py.read_pnml(path_petrinet)
decision_points = retrive_XOR_petrinet(path_petrinet)
parameters = {
    alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
aligned_log_train = []
for trace in train_log:
    new_trace = []
    caseid = trace.attributes['concept:name']
    trace_log = pm4py.filter_trace_attribute_values(train_log, 'concept:name', [caseid], retain=True)
    aligned_traces = alignments.apply_log(trace_log, net, im, fm, parameters=parameters)
    new_trace = extract_enriched_alignment_last(trace_log[0], aligned_traces, ATTRIB)
    aligned_log_train.append(new_trace)
print('TRAIN LOG aligned')
aligned_log_test = []
for trace in test_log:
    new_trace = []
    caseid = trace.attributes['concept:name']
    trace_log = pm4py.filter_trace_attribute_values(test_log, 'concept:name', [caseid], retain=True)
    aligned_traces = alignments.apply_log(trace_log, net, im, fm, parameters=parameters)
    new_trace = extract_enriched_alignment_last(trace_log[0], aligned_traces, ATTRIB)
    aligned_log_test.append(new_trace)
print('TRAIN LOG aligned')

print('######################## LOG aligned ########################')
data_to_save = {}
data_to_save['ATTRIB'] = ATTRIB
transitions_petri = [trans.name for trans in net.transitions] + ['PAD']
activity2n = {a: n for n, a in enumerate(transitions_petri)}
n2activity = {n: a for n, a in enumerate(transitions_petri)}
data_to_save['encoding_transitions'] = activity2n
print('decision_points', decision_points)
for decision in decision_points:
    print("############## Compute training for", decision, "decision point ##############")
    list_transitions = decision_points[decision]
    trace_filter_train = trace_filter_decision_point(aligned_log_train, list_transitions)
    prefix_traces_train = create_prefix(trace_filter_train, list_transitions)

    trace_filter_test = trace_filter_decision_point(aligned_log_test, list_transitions)
    prefix_traces_test = create_prefix(trace_filter_test, list_transitions)

    max_prefix = max(find_max_length(prefix_traces_train), find_max_length(prefix_traces_test))

    df_train = latest_payload_encoding(prefix_traces_train, ATTRIB, activity2n, max_prefix)
    df_test = latest_payload_encoding(prefix_traces_test, ATTRIB, activity2n, max_prefix)
    data_to_save[decision] = {}
    data_to_save[decision]['transitions'] = list_transitions
    data_to_save[decision]['padding'] = max_prefix

    y_train = df_train.label  # Target variable
    X_train = df_train.drop(['label'], axis=1)
    y_test = df_test.label  # Target variable
    X_test = df_test.drop(['label'], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    space = {'max_depth': scope.int(hp.quniform('max_depth', 1, 21, 1)),
             'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 11)),
             'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 3, 26)),
             'criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss'])}

    criterion_dict = {0: 'gini', 1: 'entropy', 2: 'log_loss'}


    def objective(params):
        clf = DecisionTreeClassifier(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        return -f1_score(y_val, y_pred, average='macro')


    if len(X_test) > 0:
        # Perform hyperparameter optimization
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials)

        # Print the best hyperparameters found
        print("Best Hyperparameters:", best)

        # model = best_candidate['model']
        # Train the model with the best hyperparameters
        # best_params = space_eval(space, best)
        clf = DecisionTreeClassifier(criterion=criterion_dict[best['criterion']],
                                          max_depth=int(best['max_depth']),
                                          min_samples_leaf=int(best['min_samples_leaf']),
                                          min_samples_split=int(best['min_samples_split']))
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        data_to_save[decision]['Accuracy'] = metrics.accuracy_score(y_test, y_pred)
        data_to_save[decision]['f1'] = f1_score(y_test, y_pred, average=None).tolist()
        data_to_save[decision]['f1_score_macro'] = f1_score(y_test, y_pred, average='macro')
        data_to_save[decision]['f1_score_micro'] = f1_score(y_test, y_pred, average='micro')
        data_to_save[decision]['f1_score_weighted'] = f1_score(y_test, y_pred, average='weighted')
        data_to_save[decision]['confusion_matrix'] = confusion_matrix(y_test, np.array(y_pred)).tolist()
        data_to_save[decision]['features_importances'] ={ list(df_train.columns)[i]: v for i, v in enumerate(clf.feature_importances_)}
    if len(X_test) > 0 and data_to_save[decision]['f1_score_macro'] > 0.60:
        pickle.dump(clf, open(decision + '.pkl', 'wb'))
        data_to_save[decision]['prediction'] = True
    else:
        data_to_save[decision]['prediction'] = False
        ### compute probability
        data_to_save[decision]['probability'] = {}
        tot = sum([len(prefix_traces_train[key]) for key in prefix_traces_train])
        for key in prefix_traces_train:
            data_to_save[decision]['probability'][key] = len(prefix_traces_train[key]) / tot

with open(NAME + "_decision_points_last_payload.json", "w") as outfile:
    json.dump(data_to_save, outfile, indent=len(data_to_save))

#########