let mouseMoved = false
let mouseTrajectory = []
let responseId = "${e://Field/ResponseID}"
let qualtricsIp = "${loc://IPAddress}" 

Qualtrics.SurveyEngine.addOnload(function()
{
    /*Place your JavaScript here to run when the page loads*/
    let qid = this.questionId
    sendIP(qualtricsIp, responseId)
    recordData(responseId, qid)
});

keepMouseTrajectory(mouseTrajectory);

Qualtrics.SurveyEngine.addOnReady(function()
{
    /*Place your JavaScript here to run when the page is fully displayed*/
    jQuery("body").mousemove(function() {
        mouseMoved = true;
    })

});

Qualtrics.SurveyEngine.addOnPageSubmit(function(type)
{
    /*Place your JavaScript here to run when the page is unloaded*/
	let qid = this.questionId
	sendEmbeddedData(responseId, "mouseTrajectory", mouseTrajectory, qid)
    appendEmbeddedData("mouseMoved", mouseMoved.toString(), qid)
	appendEmbeddedData("mouseTrajectory", JSON.stringify(mouseTrajectory), qid)

});