## Opening a Pull Request:
Following these guidelines will increase the likelihood of your pull request being accepted:

* Before pushing your PR to the repository, make sure that it builds perfectly fine on your local system.
* Add enough information, like a meaningful title, the reason why you made the commit and a link to the issue page if you opened one for this PR.
* Scope your PR to one issue. Before submitting, make sure the diff contains no unrelated changes. If you want to cover more than one issue, submit your changes for each as separate pull requests.
* Try not to include "oops" commits - ones that just fix an error in the previous commit. If you have those, then before submitting squash those fixes directly into the commits where they belong.
* Strictly Follow: **One Issue == One Pull Request == One Commit**

## Testing and merging pull requests
Your pull request will be automatically tested by Travis CI. If any jobs have failed, you should fix them. 
To rerun the automatic builds just push changes to your branch on GitHub. No need to close that pull request and open a new one!
Once all the builders are "green", one of DetectionShuite's developers will review your code. Reviewer could ask you to modify your pull request. Please provide timely response for reviewers (within weeks, not months), otherwise you submission could be postponed or even rejected.
