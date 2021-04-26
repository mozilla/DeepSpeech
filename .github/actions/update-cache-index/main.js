const core = require("@actions/core");
const exec = require("@actions/exec");
const fs = require("fs").promises;
const github = require("@actions/github");
const {HttpClient} = require("@actions/http-client");
const {BearerCredentialHandler} = require("@actions/http-client/auth");

function createHttpClient(userAgent) {
  return new HttpClient(userAgent, [
    // From https://github.com/actions/toolkit/blob/main/packages/artifact/src/internal/config-variables.ts
    new BearerCredentialHandler(process.env.ACTIONS_RUNTIME_TOKEN)
  ]);
}

async function getGoodWorkflowArtifactsFromAPI(name) {
    // From https://github.com/actions/toolkit/blob/main/packages/artifact/src/internal/config-variables.ts
    const runtimeUrl = process.env.ACTIONS_RUNTIME_URL;
    const runId = process.env.GITHUB_RUN_ID;
    // From https://github.com/actions/toolkit/blob/main/packages/artifact/src/internal/utils.ts
    const apiVersion = "6.0-preview";
    let url = `${runtimeUrl}_apis/pipelines/workflows/${runId}/artifacts?api-version=${apiVersion}`

    const client = createHttpClient("@actions/artifact-download");
    const response = await client.get(url, {
        "Content-Type": "application/json",
        "Accept": `application/json;api-version=${apiVersion}`,
    });

    const allArtifacts = JSON.parse(await response.readBody()).value;
    console.log(`==> Got ${allArtifacts.length} artifacts in response`);
    const goodWorkflowArtifacts = allArtifacts.filter(artifact => {
        return artifact.name === name
    });

    console.log("==> maybe goodWorkflowArtifacts:", goodWorkflowArtifacts);
    return goodWorkflowArtifacts;
}

async function main() {
    const token = core.getInput("github-token", {required: true});
    const artifactName = core.getInput("artifact-name", {required: true});
    const localCacheIndexPath = core.getInput("local-cache-index-path", {required: true});
    const branch = core.getInput("branch");

    // Get artifact ID from API and cache it
    const foundArtifacts = await getGoodWorkflowArtifactsFromAPI(artifactName);
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
