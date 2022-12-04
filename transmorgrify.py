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

FILE_VERSION = 1

class Transmorgrifier:
    def train( self, from_sentences, to_sentences, iterations = 4000, device = 'cpu', trailing_context = 7, leading_context = 7, verbose=True ):
        """
        Train the Transmorgrifier model.  This does not save it to disk but just trains in memory.

        Keyword arguments:
        from_sentences -- An array of strings for the input sentences.
        to_sentences -- An array of strings of the same length as from_sentences which the model is to train to convert to.
        iterations -- An integer specifying the number of iterations to convert from or to. (default 4000)
        device -- The gpu reference which catboost wants or "cpu". (default cpu)
        trailing_context -- The number of characters after the action point to include for context. (default 7)
        leading_context -- The number of characters before the action point to include for context. (default 7)
        verbose -- Increased the amount of text output during training. (default True)
        """
        X,Y = _parse_for_training( from_sentences, to_sentences, num_pre_context_chars=leading_context, num_post_context_chars=trailing_context )

        #train and save the action_model
        self.action_model = _train_catboost( X, Y['action'], iterations, verbose=verbose, device=device, model_piece='action' )

        #and the char model
        #slice through where only the action is insert.
        insert_indexes = Y['action'] == INSERT_TO
        self.char_model = _train_catboost( X[insert_indexes], Y['char'][insert_indexes], iterations, verbose=verbose, device=device, model_piece='char' )

        self.trailing_context = trailing_context
        self.leading_context = leading_context
        self.iterations = iterations

        return self

    def save( self, model='my_model.tm' ):
        """
        Saves the model previously trained with train to a specified model file.
        
        Keyword arguments:
        model -- The pathname to save the model such as "my_model.tm" (default my_model.tm)
        """
        self.name = model
        with zipfile.ZipFile( model, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9 ) as myzip:
            with myzip.open( 'params.json', mode='w' ) as out:
                out.write( json.dumps({
                    'version': FILE_VERSION,
                    'leading_context': self.leading_context,
                    'trailing_context': self.trailing_context,
                    'iterations': self.iterations,
                }).encode())
            temp_filename = _mktemp()
            self.action_model.save_model( temp_filename )
            myzip.write( temp_filename, "action.cb" )
            self.char_model.save_model( temp_filename )
            myzip.write( temp_filename,   "char.cb" )
            os.unlink( temp_filename )

        return self

    def load( self, model='my_model.tm' ):
        """
        Loads the model previously saved from the file system.

        Keyword arguments:
        model -- The filename of the model to load. (default my_model.tm)
        """
        self.name = model
        with zipfile.ZipFile( model, mode='r' ) as zip:
            with zip.open( 'params.json' ) as fin:
                params = json.loads( fin.read().decode() )
                if params['version'] > FILE_VERSION: raise Exception( f"Version {params['version']} greater than {FILE_VERSION}" )
                self.leading_context = int(params['leading_context'])
                self.trailing_context = int(params['trailing_context'])
                self.iterations = int(params['iterations'])
            temp_filename = _mktemp()
            with zip.open( 'action.cb' ) as fin:
                with open( temp_filename, "wb" ) as fout:
                    fout.write( fin.read() )
            self.action_model = CatBoostClassifier().load_model( temp_filename )
            with zip.open( 'char.cb' ) as fin:
                with open( temp_filename, "wb" ) as fout:
                    fout.write( fin.read() )
            self.char_model   = CatBoostClassifier().load_model(  temp_filename   )

        os.unlink( temp_filename)

        return self

    
    def execute( self, from_sentences, verbose=False ):
        """
        Runs the data from from_sentaces.  The results are returned 
        using yield so you need to wrap this in list() if you want 
        to index it.  from_sentences can be an array or a generator.

        Keyword arguments:
        from_sentences -- Something iterable which returns strings.
        """
        for i,from_sentence in enumerate(from_sentences):

            yield _do_reconstruct( 
                action_model=self.action_model, 
                char_model=self.char_model, 
                text=from_sentence, 
                num_pre_context_chars=self.leading_context, 
                num_post_context_chars=self.trailing_context  )
            if verbose and i % 10 == 0:
                print( f"{i} of {len(from_sentences)}" )

    def demo( self, share=False ):
        import gradio as gr 

        def gradio_function( text ):
            return list(self.execute( [text] ))[0]

        with gr.Blocks() as demo:
            name = gr.Markdown( self.name )
            inp = gr.Textbox( label="Input" )
            out = gr.Textbox( label="Output" )
            inp.change( gradio_function, inputs=[inp], outputs=[out] )
        demo.launch( share=share )

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

def _trace_edits( from_sentence, to_sentence, print_debug=False ):
    #iterating from will be the rows down the left side.
    #iterating to will be the columns across the top.
    #we will keep one row as we work on the next.

    last_row = None
    current_row = []

    #the index handles one before the index in the string
    #to handle the root cases across the top and down the left of the
    #match matrix.
    for from_row_i in range( len(from_sentence)+1 ):

        for to_column_i in range( len(to_sentence )+1 ):

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
                    best_option.char = to_sentence[to_column_i-1]
                    best_option.from_row_i = from_row_i
                    best_option.to_column_i = to_column_i
                    best_option.action = INSERT_TO
            
            #check up
            if from_row_i > 0:
                if best_option is None or last_row[to_column_i].edit_distance + 1 < best_option.edit_distance:
                    best_option = _edit_trace_hop()
                    best_option.parrent = last_row[to_column_i]
                    best_option.edit_distance = best_option.parrent.edit_distance + 1
                    best_option.char = from_sentence[from_row_i-1]
                    best_option.from_row_i = from_row_i
                    best_option.to_column_i = to_column_i
                    best_option.action = DELETE_FROM

                #check match
                if to_column_i > 0:
                    if to_sentence[to_column_i-1] == from_sentence[from_row_i-1]:
                        if best_option is None or last_row[to_column_i-1].edit_distance <= best_option.edit_distance: #prefer match so use <= than <
                            best_option = _edit_trace_hop()
                            best_option.parrent = last_row[to_column_i-1]
                            best_option.edit_distance = best_option.parrent.edit_distance + 1
                            best_option.char = from_sentence[from_row_i-1]
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


def _parse_single_for_training( from_sentence, to_sentence, num_pre_context_chars, num_post_context_chars ):
    trace = _trace_edits( from_sentence, to_sentence )

    #we will collect a snapshot at each step.
    trace_list = _list_trace(trace)


    training_collection = []

    #execute these things on the from_sentence and see if we get the to_sentence.
    working_from = from_sentence
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

    
    if to_sentence != working_to:
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


def _parse_for_training( from_sentences, to_sentences, num_pre_context_chars, num_post_context_chars ):
    out_observations_list = []
    out_results_list = []

    for index, (from_sentence, to_sentence) in enumerate(zip( from_sentences, to_sentences )):
        if type(from_sentence) != float and type(to_sentence) != float: #bad lines are nan which are floats.
            specific_observation, specific_result = _parse_single_for_training( from_sentence, to_sentence, num_pre_context_chars=num_pre_context_chars, num_post_context_chars=num_post_context_chars )

            out_observations_list.append( specific_observation )
            out_results_list.append( specific_result )
        if index % 100 == 0:
            print( f"parsing {index} of {len(from_sentences)}")

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

    if( verbose ): print( '{} is fitted: {}'.format(model_piece,model.is_fitted()))
    if( verbose ): print( '{} params:\n{}'.format(model_piece,model.get_params()))

    return model



def _mktemp():
    #I know mktemp exists in the library but it has been depricated suggesting using
    #mkstemp but catboost can't write to a filehandle yet, so I need an actual
    #filename.
    number = 0
    while os.path.exists( f".temp_{number}~" ):
        number += 1
    return f".temp_{number}~"


def _do_reconstruct( action_model, char_model, text, num_pre_context_chars, num_post_context_chars  ):
    # result = ""
    # for i in range(len(text)):
    #     pre_context = ( (" " * num_pre_context_chars) + result[max(0,len(result)-num_pre_context_chars):])[-num_pre_context_chars:]
    #     post_context = (text[i:min(len(text),i+num_post_context_chars)] + (" " * num_post_context_chars))[:num_post_context_chars]
    #     full_context = pre_context + post_context
    #     context_as_dictionary = { 'c'+str(c):[full_context[c]] for c in range(len(full_context)) }
    #     context_as_pd = pd.DataFrame( context_as_dictionary )

    #     model_result = model.predict( context_as_pd )[0]

    #     if not quite and len( result ) % 500 == 0: print( "%" + str(i*100/len(text))[:4] + " " + result[-100:])

    #     if model_result: result += " "
    #     result += text[i]

    #     pass
    # return result

    #test for nan.
    if text != text: text = ''

    working_from = text
    working_to = ""
    used_from = ""
    continuous_added = 0
    continuous_dropped = 0
    while working_from and len(working_to) < 3*len(text) and (len(working_to) < 5 or working_to[-5:] != (working_to[-1] * 5)):
        from_context = (working_from + (" " * num_post_context_chars))[:num_post_context_chars]
        to_context =   ((" " * num_pre_context_chars) + working_to )[-num_pre_context_chars:]
        used_context = ((" " * num_pre_context_chars) + used_from  )[-num_pre_context_chars:]

        #construct the context.
        context_as_dictionary = {}
        #from_context
        for i in range( num_post_context_chars ):
            context_as_dictionary[ f"f{i}" ] = [from_context[i]]
        #to_context
        for i in range( num_pre_context_chars ):
            context_as_dictionary[ f"t{i}" ] = [to_context[i]]
        #used_context
        for i in range( num_pre_context_chars ):
            context_as_dictionary[ f"u{i}" ] = [used_context[i]]
        #these two things.
        context_as_dictionary["continuous_added"]   = [continuous_added]
        context_as_dictionary["continuous_dropped"] = [continuous_dropped]

        #make it a pandas.
        context_as_pd = pd.DataFrame( context_as_dictionary )

        #run the model
        action_model_result = action_model.predict( context_as_pd )[0][0]

        if action_model_result == START:
            pass
        elif action_model_result == INSERT_TO:
            #for an insert ask the char model what to insert
            char_model_result = char_model.predict( context_as_pd )[0][0]

            working_to += char_model_result
            continuous_added += 1
            continuous_dropped = 0
        elif action_model_result == DELETE_FROM:
            used_from += working_from[0]
            working_from = working_from[1:]
            continuous_added = 0
            continuous_dropped += 1
        elif action_model_result == MATCH:
            used_from += working_from[0]
            working_to += working_from[0]
            working_from = working_from[1:]
            continuous_added = 0
            continuous_dropped = 0

    return working_to


#edit distance from https://stackoverflow.com/a/32558749/1419054
def _levenshteinDistance(s1, s2):
    if s1 != s1: s1 = ''
    if s2 != s2: s2 = ''
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def train( in_csv, a_header, b_header, model, iterations, device, leading_context, trailing_context, train_percentage, verbose ):
    if verbose: print( "loading csv" )
    full_data = pd.read_csv( in_csv )

    split_index = int( train_percentage/100*len(full_data) )
    train_data = full_data.iloc[:split_index,:].reset_index(drop=True)

    if verbose: print( "parcing data for training" )


    tm = Transmorgrifier()

    tm.train( from_sentences=train_data[a_header], 
            to_sentences=train_data[b_header], 
            iterations = iterations,
            device = device,
            leading_context = leading_context, 
            trailing_context = trailing_context,
            verbose=verbose,
                )
    tm.save( model )

def execute( include_stats, in_csv, out_csv, a_header, b_header, model, execute_percentage, verbose ):
    if verbose: print( "loading csv" )

    full_data = pd.read_csv( in_csv )

    split_index = int( (100-execute_percentage)/100*len(full_data) )
    execute_data = full_data.iloc[split_index:,:].reset_index(drop=True)


    tm = Transmorgrifier()
    tm.load( model )

    results = list(tm.execute( execute_data[a_header ], verbose=verbose ))

    
    if include_stats:
        before_edit_distances = []
        after_edit_distances = []
        percent_improvement = []

        for row in range(len( execute_data )):
            before_edit_distances.append(
                _levenshteinDistance( execute_data[a_header][row], execute_data[b_header][row] )
            )
            after_edit_distances.append(
                _levenshteinDistance( results[row], execute_data[b_header][row] )
            )
            percent_improvement.append(
                100*(before_edit_distances[row] - after_edit_distances[row])/max(1,before_edit_distances[row])
            )

        pd_results = pd.DataFrame( {
            "in_data": execute_data[a_header],
            "out_data": execute_data[b_header],
            "generated_data": results,
            "before_edit_distance": before_edit_distances,
            "after_edit_distance": after_edit_distances,
            "percent_improvement": percent_improvement,
        })
        pd_results.to_csv( out_csv )
    else:
        pd_results = pd.DataFrame( {
            "out_data": execute_data[b_header],
        })
        pd_results.to_csv( out_csv )

def safe_float( str ):
    if str is not None:
        return float(str)
    return None #explicit None return.
    
def main():
    parser = argparse.ArgumentParser(
                    prog = 'transmorgrify.py',
                    description = 'Converts text from one to another according to a model.',
                    epilog = '(C) Joshua Lansford')
    parser.add_argument('-t', '--train', action='store_true', help='Train a model instead of executing a model')
    parser.add_argument('-e', '--execute', action='store_true', help='Use an existing trained model.')
    parser.add_argument('-g', '--gradio', action='store_true', help='Start a gradio demo with the selected model.' )
    parser.add_argument('-s', '--share', action='store_true', help="Share the gradio app with a temporary public URL." )
    parser.add_argument('-i', '--in_csv',  help='The csv to read training or input data from', default='in.csv' )     
    parser.add_argument('-o', '--out_csv',  help='The csv to write conversion to', default='out.csv' )     
    parser.add_argument('-a', '--a_header', help='The column header for training or transforming from', default="source" )
    parser.add_argument('-b', '--b_header',   help='The column header for training the transformation to', default="target"  )
    parser.add_argument('-m', '--model',help='The model file to create during training or use during transformation', default='model.tm' )
    parser.add_argument('-n', '--iterations', help='The number of iterations to train', default=2000 )
    parser.add_argument('-d', '--device',  help='Which device, i.e. if useing GPU', default='cpu' )
    parser.add_argument('-x', '--context', help='The number of leading and trailing chars to use as context', default=7 )
    parser.add_argument('-p', '--train_percentage', help="The percentage of data to train on, leaving the rest for testing.")
    parser.add_argument('-v', '--verbose', action='store_true', help='Talks alot?' )
    parser.add_argument('-c', '--include_stats',   action='store_true', help='Use b_header to compute stats and add to output csv.')
                        

    args = parser.parse_args()

    if not args.train and not args.execute and not args.gradio: print( "Must include --execute, --train and/or --gradio to do something." )

    
    if args.train:
        train_percentage = safe_float(args.train_percentage)
        if train_percentage is None:
            if args.execute:
                train_percentage = 50
            else:
                train_percentage = 100

        train( in_csv=args.in_csv, 
               a_header=args.a_header, 
               b_header=args.b_header, 
               model=args.model,
               iterations=int(args.iterations),
               device=args.device,
               leading_context=int(args.context),
               trailing_context=int(args.context),
               train_percentage=train_percentage,
               verbose=args.verbose,
               )


    if args.execute:
        if args.train_percentage is None:
            if args.train:
                execute_percentage = 50
            else:
                execute_percentage = 100
        else:
            execute_percentage = 100-safe_float(args.train_percentage)
        execute(
            include_stats=args.include_stats,
            in_csv=args.in_csv, 
            out_csv=args.out_csv, 
            a_header=args.a_header, 
            b_header=args.b_header, 
            model=args.model, 
            execute_percentage=execute_percentage, 
            verbose=args.verbose,
        )


    if args.gradio:
        tm = Transmorgrifier()
        tm.load( args.model )

        tm.demo( args.share is not None )


if __name__ == '__main__':
    main()
