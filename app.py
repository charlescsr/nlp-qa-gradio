from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
import gradio as gr

passage = "Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens.  The Texans would respond with fullback Vonta Leach getting a 1-yard touchdown run, yet the Raiders would answer with kicker Sebastian Janikowski getting a 33-yard and a 30-yard field goal.  Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal. Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt."
question = "How many yards was the longest passing touchdown?"

def qa(passage, question):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo.2021-02-11.tar.gz")
    ans = predictor.predict(
            passage=passage,
            question=question
        )
    
    return ans['best_span_str']

iface = gr.Interface(qa, 
                     inputs=[gr.inputs.Textbox(lines=25, default=passage), 
                             gr.inputs.Textbox(default=question)],
                     outputs=["text"])
iface.launch()