const core = require('@actions/core');
const github = require('@actions/github');
const AdmZip = require('adm-zip');
const filesize = require('filesize');
const pathname = require('path');
const fs = require('fs');
const { throttling } = require('@octokit/plugin-throttling');
const { GitHub } = require('@actions/github/lib/utils');

async function getGoodArtifacts(client, owner, repo, name) {
    const goodRepoArtifacts = await client.paginate(
        "GET /repos/{owner}/{repo}/actions/artifacts",
        {
            owner: owner,
            repo: repo,
            per_page: 100,
        },
        (repoArtifacts, done) => {
            // console.log(" ==> repoArtifacts", repoArtifacts);
            const goodArtifacts = repoArtifacts.data.filter((a) => {
                // console.log("==> Artifact check", a);
                return a.name == name
            });
            if (goodArtifacts.length > 0) {
                done();
            }
            return goodArtifacts;
        }
    );

    console.log("==> maybe goodRepoArtifacts:", goodRepoArtifacts);
    return goodRepoArtifacts;
}

async function main() {
    const token = core.getInput("github_token", { required: true });
    const [owner, repo] = core.getInput("repo", { required: true }).split("/");
    const path = core.getInput("path", { required: true });
    const name = core.getInput("name");
    const download = core.getInput("download");
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

    const goodArtifacts = await getGoodArtifacts(client, owner, repo, name);
    console.log("==> goodArtifacts:", goodArtifacts);

    let artifactStatus = "";
        if (goodArtifacts.length === 0) {
        artifactStatus = "missing";
    } else {
        artifactStatus = "found";
    }

    console.log("==> Artifact", name, artifactStatus);
    console.log("==> download", download);

    core.setOutput("status", artifactStatus);

    if (artifactStatus === "found" && download == "true") {
        console.log("==> # artifacts:", goodArtifacts.length);

        let artifact = goodArtifacts[0];

        console.log("==> Artifact:", artifact.id)

        const size = filesize(artifact.size_in_bytes, { base: 10 })

        console.log("==> Downloading:", artifact.name + ".zip", `(${size})`)

        const zip = await client.actions.downloadArtifact({
            owner: owner,
            repo: repo,
            artifact_id: artifact.id,
            archive_format: "zip",
        })

        const dir = name ? path : pathname.join(path, artifact.name)

        fs.mkdirSync(dir, { recursive: true })

        const adm = new AdmZip(Buffer.from(zip.data))

        adm.getEntries().forEach((entry) => {
            const action = entry.isDirectory ? "creating" : "inflating"
            const filepath = pathname.join(dir, entry.entryName)
            console.log(`  ${action}: ${filepath}`)
        })

        adm.extractAllTo(dir, true)
    }

    if (artifactStatus === "missing" && download == "true") {
        core.setFailed("Required", name, "that is missing");
    }

    return;
}

// We have to manually wrap the main function with a try-catch here because
// GitHub will ignore uncatched exceptions and continue running the workflow,
// leading to harder to diagnose errors downstream from this action.
try {
    main();
} catch (error) {
    core.setFailed(error.message);
}
