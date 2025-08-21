from flask import Flask, render_template, request
from main import determine_risk_profile, recommend_investment_options, calculate_prediction_with_chart

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    step = 1
    recommendation = None
    investment_options = []
    prediction = None
    chart_img = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form.to_dict()

        if 'investment_type' not in form_data:
            age = int(form_data['age'])
            horizon = int(form_data['horizon'])
            risk = form_data['risk']
            recommendation = determine_risk_profile(age, horizon, risk)
            investment_options = recommend_investment_options(recommendation)
            step = 2
        else:
            option = form_data['investment_type']
            monthly_amount = float(form_data['amount'])
            horizon = int(form_data['horizon'])
            prediction, chart_img = calculate_prediction_with_chart(option, monthly_amount, horizon)
            step = 3

    return render_template('index.html',
                           step=step,
                           recommendation=recommendation,
                           investment_options=investment_options,
                           form_data=form_data,
                           prediction=prediction,
                           chart_img=chart_img)

if __name__ == '__main__':
    app.run(debug=True)
