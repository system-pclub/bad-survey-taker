/** Q195 */

Qualtrics.SurveyEngine.addOnload(function()
{
	/*Place your JavaScript here to run when the page loads*/
    this.hideNextButton()
	
});

Qualtrics.SurveyEngine.addOnReady(function()
{
	/*Place your JavaScript here to run when the page is fully displayed*/
	setTimeout(function() {
		  if (confirm('Please click "OK" to continue')) {
		      jQuery("#NextButton").click()
		  }
		  }, 5 * 1000)
});

Qualtrics.SurveyEngine.addOnUnload(function()
{
	/*Place your JavaScript here to run when the page is unloaded*/

});