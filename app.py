import gradio as gr
import transmorgrify

eng_to_ipa_tm = transmorgrify.Transmorgrifier()
eng_to_ipa_tm.load( "./examples/phonetic/phonetics_gpu_4000.tm" )

ipa_to_eng_tm = transmorgrify.Transmorgrifier()
ipa_to_eng_tm.load( "./examples/phonetic/reverse_phonetics_gpu_4000.tm")

eng_to_pig_tm = transmorgrify.Transmorgrifier()
eng_to_pig_tm.load( "./examples/piglattin/piglattin_gpu_4000.tm" )

pig_to_eng_tm = transmorgrify.Transmorgrifier()
pig_to_eng_tm.load( "./examples/piglattin/reverse_piglattin_gpu_4000.tm" )


def eng_to_ipa( input ):
    return list(eng_to_ipa_tm.execute( [input] ) )[0]

def ipa_to_eng( input ):
    return list(ipa_to_eng_tm.execute( [input] ) )[0]

def eng_to_pig( input ):
    return list(eng_to_pig_tm.execute( [input] ) )[0]

def pig_to_eng( input ):
    return list(pig_to_eng_tm.execute( [input] ) )[0]

with gr.Blocks() as demo:
    gr.Markdown(
"""
# Sentance Transmorgrifier demo
The following demos have been trained on different tasks.
Select the tab below for a demo.
"""
    )

    with gr.Tab( "IPA" ):
        english_in = gr.Textbox( label="English in" )
        ipa_out = gr.Textbox( label='IPA out')
        gr.Button( value='Transmorgrify' ).click( eng_to_ipa, english_in, ipa_out )

        ipa_in = gr.Textbox( label="IPA in" )
        english_out = gr.Textbox( label='English out')
        gr.Button( value='Transmorgrify' ).click( ipa_to_eng , ipa_in, english_out )

    with gr.Tab( "Piglattin" ):
        english_in = gr.Textbox( label="English in" )
        pig_out = gr.Textbox( label='Pig latin out')
        gr.Button( value='Transmorgrify' ).click( eng_to_pig, english_in, pig_out )

        pig_in = gr.Textbox( label="Pig latin in" )
        english_out = gr.Textbox( label='English out')
        gr.Button( value='Transmorgrify' ).click( pig_to_eng , pig_in, english_out )
demo.launch()
