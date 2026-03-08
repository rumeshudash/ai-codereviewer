import { readFileSync } from "fs";
import * as core from "@actions/core";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { generateText, Output } from "ai";
import { Octokit } from "@octokit/rest";
import minimatch from "minimatch";
import parseDiff, { Chunk, File } from "parse-diff";
import { z } from "zod";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");
const GEMINI_API_KEY: string = core.getInput("GEMINI_API_KEY");
const GEMINI_MODEL: string = core.getInput("GEMINI_MODEL");

const openaiProvider = OPENAI_API_KEY
  ? createOpenAI({ apiKey: OPENAI_API_KEY })
  : null;
const googleProvider = GEMINI_API_KEY
  ? createGoogleGenerativeAI({ apiKey: GEMINI_API_KEY })
  : null;

const model = openaiProvider
  ? openaiProvider(OPENAI_API_MODEL)
  : googleProvider
  ? googleProvider(GEMINI_MODEL)
  : (() => {
      core.setFailed(
        "Either OPENAI_API_KEY or GEMINI_API_KEY must be set. Provide one to run the code review."
      );
      process.exit(1);
    })();

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const reviewOutputSchema = z.object({
  reviews: z
    .array(
      z.object({
        lineNumber: z.union([z.string(), z.number()]),
        endLineNumber: z.union([z.string(), z.number()]).optional(),
        reviewComment: z.string(),
      })
    )
    .optional(),
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

type ReviewCommentInput = {
  body: string;
  path: string;
  line: number;
  side: "LEFT" | "RIGHT";
  start_line?: number;
  start_side?: "LEFT" | "RIGHT";
};

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<ReviewCommentInput[]> {
  const comments: ReviewCommentInput[] = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue; // Ignore deleted files
    for (const chunk of file.chunks) {
      const prompt = createPrompt(file, chunk, prDetails);
      console.log("Prompt:", prompt);
      const aiResponse = await getAIResponse(prompt);
      console.log("AI Response:", aiResponse);
      if (aiResponse) {
        const newComments = createComment(file, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
    }
  }
  return comments;
}

function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails): string {
  return `Your task is to review pull requests. Instructions:
- Provide the response in following JSON format: {"reviews": [{"lineNumber": <line>, "endLineNumber": <optional_end_line>, "reviewComment": "<comment>"}]}
- Use lineNumber for the primary line to comment on (or the last line of a range). Use endLineNumber only when the comment applies to a contiguous range of lines (first line of range); omit it for single-line comments.
- Do not give positive comments or compliments.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Use the given description only for the overall context and only comment the code.
- IMPORTANT: NEVER suggest adding comments to the code.

Review the following code diff in the file "${
    file.to
  }" and take the pull request title and description into account when writing the response.
  
Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes
  // @ts-expect-error - ln and ln2 exists where needed
  .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join("\n")}
\`\`\`
`;
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  endLineNumber?: string;
  reviewComment: string;
}> | null> {
  try {
    const { output } = await generateText({
      model,
      system: prompt,
      prompt: "",
      maxOutputTokens: 700,
      temperature: 0.2,
      output: Output.object({
        schema: reviewOutputSchema,
        name: "CodeReview",
        description: "Code review comments for a diff chunk",
      }),
    });

    const reviews =
      output.reviews?.map((r) => ({
        lineNumber: String(r.lineNumber),
        endLineNumber: r.endLineNumber != null ? String(r.endLineNumber) : undefined,
        reviewComment: r.reviewComment,
      })) ?? null;
    return reviews;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

function createComment(
  file: File,
  aiResponses: Array<{
    lineNumber: string;
    endLineNumber?: string;
    reviewComment: string;
  }>
): ReviewCommentInput[] {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }
    const line = Number(aiResponse.lineNumber);
    const endLine =
      aiResponse.endLineNumber != null
        ? Number(aiResponse.endLineNumber)
        : undefined;
    const isMultiLine =
      endLine != null && endLine !== line;
    const [startLine, lastLine] =
      isMultiLine && endLine != null
        ? endLine < line
          ? [endLine, line]
          : [line, endLine]
        : [line, line];

    const comment: ReviewCommentInput = {
      body: aiResponse.reviewComment,
      path: file.to,
      line: lastLine,
      side: "RIGHT",
    };
    if (isMultiLine) {
      comment.start_line = startLine;
      comment.start_side = "RIGHT";
    }
    return comment;
  });
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: ReviewCommentInput[]
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    event: "COMMENT",
  });
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
