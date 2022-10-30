#!/usr/bin/env python3 
import argparse
import json
import os
import zipfile

import pandas as pd
from catboost import CatBoostClassifier, Pool

MATCH = 0
DELETE_FROM = 1
INSERT_TO = 2
START = 3



def _list_trace( trace ):
    if trace.parrent is None:
        result = [trace]
    else:
        result = _list_trace( trace.parrent )
        result.append( trace )
    return result

class _edit_trace_hop():
    parrent = None
    edit_distance = None
    char = None
    from_row_i = None
    to_column_i = None
    action = None

    def __str__( self ):
        if self.action == START:
            return "<start>"
        elif self.action == INSERT_TO:
            return f"<ins> {self.char}"
        elif self.action == DELETE_FROM:
            return f"<del> {self.char}"
        elif self.action == MATCH:
            return f"<match> {self.char}"
        return "eh?"

    def __repr__( self ):
        return self.__str__()

def _trace_edits( from_sentance, to_sentance, print_debug=False ):
    #iterating from will be the rows down the left side.
    #iterating to will be the columns across the top.
    #we will keep one row as we work on the next.

    last_row = None
    current_row = []

    #the index handles one before the index in the string
    #to handle the root cases across the top and down the left of the
    #match matrix.
    for from_row_i in range( len(from_sentance)+1 ):

        for to_column_i in range( len(to_sentance )+1 ):

            best_option = None

            #root case.
            if from_row_i == 0 and to_column_i == 0:
                best_option = _edit_trace_hop()
                best_option.parrent = None
                best_option.edit_distance = 0
                best_option.char = ""
                best_option.from_row_i = from_row_i
                best_option.to_column_i = to_column_i
                best_option.action = START

            #check left
            if to_column_i > 0:
                if best_option is None or current_row[to_column_i-1].edit_distance + 1 < best_option.edit_distance:
                    best_option = _edit_trace_hop()
                    best_option.parrent = current_row[to_column_i-1]
                    best_option.edit_distance = best_option.parrent.edit_distance + 1
                    best_option.char = to_sentance[to_column_i-1]
                    best_option.from_row_i = from_row_i
                    best_option.to_column_i = to_column_i
                    best_option.action = INSERT_TO
            
            #check up
            if from_row_i > 0:
                if best_option is None or last_row[to_column_i].edit_distance + 1 < best_option.edit_distance:
                    best_option = _edit_trace_hop()
                    best_option.parrent = last_row[to_column_i]
                    best_option.edit_distance = best_option.parrent.edit_distance + 1
                    best_option.char = from_sentance[from_row_i-1]
                    best_option.from_row_i = from_row_i
                    best_option.to_column_i = to_column_i
                    best_option.action = DELETE_FROM

                #check match
                if to_column_i > 0:
                    if to_sentance[to_column_i-1] == from_sentance[from_row_i-1]:
                        if best_option is None or last_row[to_column_i-1].edit_distance <= best_option.edit_distance: #prefer match so use <= than <
                            best_option = _edit_trace_hop()
                            best_option.parrent = last_row[to_column_i-1]
                            best_option.edit_distance = best_option.parrent.edit_distance + 1
                            best_option.char = from_sentance[from_row_i-1]
                            best_option.from_row_i = from_row_i
                            best_option.to_column_i = to_column_i
                            best_option.action = MATCH

            if best_option is None: raise Exception( "Shouldn't end up with best_option being None" )
            current_row.append(best_option)

        last_row = current_row
        current_row = []

    if print_debug:
        def print_diffs( current_node ):
            if current_node.parrent is not None:
                print_diffs( current_node.parrent )
            
            if current_node.action == START:
                print( "start" )
            elif current_node.action == MATCH:
                print( f"match {current_node.char}" )
            elif current_node.action == INSERT_TO:
                print( f"insert {current_node.char}" )
            elif current_node.action == DELETE_FROM:
                print( f"del {current_node.char}" )
        print_diffs( last_row[-1] )
    return last_row[-1]


def _parse_single_for_training( from_sentance, to_sentance, num_pre_context_chars, num_post_context_chars ):
    trace = _trace_edits( from_sentance, to_sentance )

    #we will collect a snapshot at each step.
    trace_list = _list_trace(trace)


    training_collection = []

    #execute these things on the from_sentance and see if we get the to_sentance.
    working_from = from_sentance
    working_to = ""
    used_from = ""
    continuous_added = 0
    continuous_dropped = 0
    for thing in trace_list:
        #gather action and context for training
        if thing.action != START:
            from_context = (working_from + (" " * num_post_context_chars))[:num_post_context_chars]
            to_context =   ((" " * num_pre_context_chars) + working_to )[-num_pre_context_chars:]
            used_context = ((" " * num_pre_context_chars) + used_from  )[-num_pre_context_chars:]

            training_collection.append({
                "from_context": from_context,
                "to_context": to_context,
                "used_context": used_context,
                "action": thing.action,
                "continuous_added": continuous_added,
                "continuous_dropped": continuous_dropped,
                "char": thing.char if thing.action == INSERT_TO else ' ',
            })

        #now execute the action for the next step.
        if thing.action == START:
            pass
        elif thing.action == INSERT_TO:
            working_to += thing.char
            continuous_added += 1
            continuous_dropped = 0
        elif thing.action == DELETE_FROM:
            used_from += working_from[0]
            working_from = working_from[1:]
            continuous_added = 0
            continuous_dropped += 1
        elif thing.action == MATCH:
            used_from += working_from[0]
            working_to += working_from[0]
            working_from = working_from[1:]
            continuous_added = 0
            continuous_dropped = 0

    
    if to_sentance != working_to:
        print( "Replay failure" )

    #so now I have training_collection which is a list of dictionaries where each dictionary is an action with a context.
    #I need to change it into a dictionary of lists where each dictionary a column and the lists are the rows.
    context_split_into_dict = {}

    #first collect the from_context:
    for i in range( num_post_context_chars ):
        this_slice = []
        for training in training_collection:
            this_slice.append( training['from_context'][i] )
        context_split_into_dict[ f"f{i}" ] = this_slice
    
    #now collect to_context:
    for i in range( num_pre_context_chars ):
        this_slice = []
        for training in training_collection:
            this_slice.append( training['to_context'][i] )
        context_split_into_dict[ f"t{i}" ] = this_slice

    #now collect used_context
    for i in range( num_pre_context_chars ):
        this_slice = []
        for training in training_collection:
            this_slice.append( training['used_context'][i] )
        context_split_into_dict[ f"u{i}" ] = this_slice

    
    #now these two things.
    context_split_into_dict["continuous_added"] = []
    context_split_into_dict["continuous_dropped"] = []
    for training in training_collection:
        context_split_into_dict["continuous_added"].append( training["continuous_added"] )
        context_split_into_dict["continuous_dropped"].append( training["continuous_dropped"] )

    #now also collect the output answers.
    result_split_into_dict = {}
    action_slice = []
    char_slice = []
    for training in training_collection:
        action_slice.append( training['action'] )
        char_slice.append( training['char'] )
    result_split_into_dict['action'] = action_slice
    result_split_into_dict['char']   = char_slice
        
    #now return it as a dataframe.
    return pd.DataFrame( context_split_into_dict ), pd.DataFrame( result_split_into_dict )


def _parse_for_training( from_sentances, to_sentances, num_pre_context_chars, num_post_context_chars ):
    out_observations_list = []
    out_results_list = []

    for index, (from_sentance, to_sentance) in enumerate(zip( from_sentances, to_sentances )):
        if type(from_sentance) != float and type(to_sentance) != float: #bad lines are nan which are floats.
            specific_observation, specific_result = _parse_single_for_training( from_sentance, to_sentance, num_pre_context_chars=num_pre_context_chars, num_post_context_chars=num_post_context_chars )

            out_observations_list.append( specific_observation )
            out_results_list.append( specific_result )
        if index % 100 == 0:
            print( f"parsing {index} of {len(from_sentances)}")

    return pd.concat( out_observations_list ), pd.concat( out_results_list )

def _train_catboost( X, y, iterations, device, verbose, model_piece, learning_rate = .07 ):

    X = X.fillna( ' ' )
    passed = False
    while not passed:
        train_pool = Pool(
            data=X,
            label=y,
            cat_features=[i for i,x in enumerate(X.keys()) if len(x) == 2] #all cat keys are length 2
        )
        validation_pool = None #Can't use validation pool because it randomly has chars not in training.
        model = CatBoostClassifier(
            iterations = iterations,
            learning_rate = learning_rate,
            task_type="GPU" if device.lower() != 'cpu' else "CPU",
            devices=device if device.lower() != 'cpu' else None
        )
        model.fit( train_pool, eval_set=validation_pool, verbose=True )
        passed = True

    if( verbose ): print( '{} is fitted: {}',format(model_piece,model.is_fitted()))
    if( verbose ): print( '{} params:\n{}'.format(model_piece,model.get_params()))

    return model

def _train_reconstruct_models( from_sentances, to_sentances, iterations, device, num_pre_context_chars, num_post_context_chars, verbose ):
  
    X,Y = _parse_for_training( from_sentances, to_sentances, num_pre_context_chars=num_pre_context_chars, num_post_context_chars=num_post_context_chars )

    #train and save the action_model
    action_model = _train_catboost( X, Y['action'], iterations, verbose=verbose, device=device, model_piece='action' )

    #and the char model
    #slice through where only the action is insert.
    insert_indexes = Y['action'] == INSERT_TO
    char_model = _train_catboost( X[insert_indexes], Y['char'][insert_indexes], iterations, verbose=verbose, device=device, model_piece='char' )

    return action_model, char_model

def _mktemp():
    #I know mktemp exists in the library but it has been depricated suggesting using
    #mkstemp but catboost can't write to a filehandle yet, so I need an actual
    #filename.
    number = 0
    while os.path.exists( f".temp_{number}~" ):
        number += 1
    return f".temp_{number}~"

def train( in_csv, a_header, b_header, model, iterations, device, leading_context,trailing_context, train_percentage, verbose ):
    if verbose: print( "loading csv" )
    full_data = pd.read_csv( in_csv )

    split_index = int( train_percentage/100*len(full_data) )
    train_data = full_data.iloc[:split_index,:].reset_index(drop=True)

    if verbose: print( "parcing data for training" )

    action_model, char_model = _train_reconstruct_models( from_sentances=train_data[a_header], 
            to_sentances=train_data[b_header], 
            iterations = iterations,
            device = device,
            num_pre_context_chars = leading_context, 
            num_post_context_chars = trailing_context,
            verbose=verbose,
                )

    temp_action_filename = _mktemp()
    action_model.save_model( temp_action_filename )
    temp_char_filename = _mktemp()
    char_model.save_model( temp_char_filename )

    with zipfile.ZipFile( model, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9 ) as myzip:
        with myzip.open( 'params.json', mode='w' ) as out:
            out.write( json.dumps({
                'version': 1,
                'leading_context': leading_context,
                'trailing_context': trailing_context,
                'iterations': iterations,
            }).encode())
        myzip.write( temp_action_filename, "action.cb" )
        myzip.write( temp_char_filename,   "char.cb" )

    os.unlink( temp_action_filename )
    os.unlink( temp_char_filename )
    
def main():
    parser = argparse.ArgumentParser(
                    prog = 'transmorgrify.py',
                    description = 'Converts text from one to another according to a model.',
                    epilog = '(C) Joshua Lansford')
    parser.add_argument('-i', '--in_csv',  help='The csv to read training or input data from', required=True )     
    parser.add_argument('-o', '--out_csv',  help='The csv to write conversion to', default='out.csv' )     
    parser.add_argument('-a', '--a_header', help='The column header for training or transforming from', default="source" )
    parser.add_argument('-b', '--b_header',   help='The column header for training the transformation to', default="target"  )
    parser.add_argument('-m', '--model',help='The model file to create during training or use during transformation', default='model.tm' )
    parser.add_argument('-n', '--iterations', help='The number of iterations to train', default=1000 )
    parser.add_argument('-d', '--device',  help='Which device, i.e. if useing GPU', default='cpu' )
    parser.add_argument('-x', '--context', help='The number of leading and trailing chars to use as context', default=7 )
    parser.add_argument('-t', '--train', action='store_true', help='Train a model instead of executing a model')
    parser.add_argument('-p', '--train_percentage', help="The percentage of data to train on, leaving the rest for testing.")
    parser.add_argument('-e', '--execute', action='store_true', help='Use an existing trained model.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Talks alot?' )
                        

    args = parser.parse_args()

    if not args.train and not args.execute: print( "Must include --execute and/or --train to do something." )

    
    if args.train:

        train_percentage = args.train_percentage
        if train_percentage is None:
            if args.execute:
                train_percentage = 50
            else:
                train_percentage = 100

        train( in_csv=args.in_csv, 
               a_header=args.a_header, 
               b_header=args.b_header, 
               model=args.model,
               iterations=args.iterations,
               device=args.device,
               leading_context=args.context,
               trailing_context=args.context,
               train_percentage=train_percentage,
               verbose=args.verbose,
               )

    #print(args)


if __name__ == '__main__':
    main()
