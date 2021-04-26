const core = require("@actions/core");
const exec = require("@actions/exec");
const github = require("@actions/github");
const fs = require("fs").promises;
const { throttling } = require("@octokit/plugin-throttling");
const { GitHub } = require("@actions/github/lib/utils");

async function getGoodWorkflowArtifactsFromAPI(client, owner, repo, name) {
    const goodWorkflowArtifacts = await client.paginate(
        "GET /repos/{owner}/{repo}/actions/runs/{run_id}/artifacts",
        {
            owner: owner,
            repo: repo,
            run_id: github.context.runId,
            per_page: 100,
        },
        (workflowArtifacts) => {
            // console.log(" ==> workflowArtifacts", workflowArtifacts);
            return workflowArtifacts.data.filter((a) => {
                // console.log("==> Artifact check", a);
                return a.name == name
            })
        }
    );

    console.log("==> maybe goodWorkflowArtifacts:", goodWorkflowArtifacts);
    if (goodWorkflowArtifacts.length > 0) {
        return goodWorkflowArtifacts;
    }

    // We have not been able to find a workflow artifact, it's really no good news
    return [];
}

async function main() {
    const token = core.getInput("github-token", {required: true});
    const [owner, repo] = core.getInput("repo", { required: true }).split("/");
    const artifactName = core.getInput("artifact-name", {required: true});
    const localCacheIndexPath = core.getInput("local-cache-index-path", {required: true});
    const branch = core.getInput("branch");
    const OctokitWithThrottling = GitHub.plugin(throttling);
    const client = new OctokitWithThrottling({
        auth: token,
        throttle: {
            onRateLimit: (retryAfter, options) => {
                console.log(
                    `Request quota exhausted for request ${options.method} ${options.url}`
                );

                // Retry twice after hitting a rate limit error, then give up
                if (options.request.retryCount <= 2) {
                    console.log(`Retrying after ${retryAfter} seconds!`);
                    return true;
                }
            },
            onAbuseLimit: (retryAfter, options) => {
                // does not retry, only logs a warning
                console.log(
                    `Abuse detected for request ${options.method} ${options.url}`
                );
            },
        },
    });
    console.log("==> Repo:", owner + "/" + repo);

    // Get artifact ID from API and cache it
    const foundArtifacts = await getGoodWorkflowArtifactsFromAPI(client, owner, repo, artifactName);
    if (foundArtifacts.length === 0) {
        const msg = "==> FATAL: Could not find artifact ID from name " + artifactName;
        core.setFailed(msg);
        throw msg;
    }

    const artifact = foundArtifacts[0];
    console.log("==> Found artifact ID = " + artifact.id);

    let index = {};
    try {
        const indexInFile = JSON.parse(await fs.readFile(localCacheIndexPath, "utf8"));
        index = indexInFile;
    } catch (error) {
        console.log("==> Failed to read index file, saving new file with single entry.");
    }

    console.log("==> Inserting into index name=" + artifactName + " -> id=" + artifact.id);
    index[artifactName] = artifact;

    console.log("==> Writing updated index file to " + localCacheIndexPath);
    await fs.writeFile(localCacheIndexPath, JSON.stringify(index));

    console.log("==> Committing and pushing");
    await exec.exec("git", ["config", "--global", "user.name", "Update Cache Index action"]);
    await exec.exec("git", ["config", "--global", "user.email", "update-cache-index-action@automated.invalid"]);
    await exec.exec("git", ["commit", "-am", "Update cache index with new artifact " + artifactName]);
    await exec.exec("git", ["push", "HEAD:" + branch]);
}

// We have to manually wrap the main function with a try-catch here because
// GitHub will ignore uncaught exceptions and continue running the workflow,
// leading to harder to diagnose errors downstream from this action.
try {
    main();
} catch (error) {
    core.setFailed(error.message);
}
