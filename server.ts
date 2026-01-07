import { GoogleGenerativeAI } from "npm:@google/generative-ai@0.21.0";

const apiKey = Deno.env.get("GEMINI_API_KEY");
if (!apiKey) {
  console.error("❌ GEMINI_API_KEY tidak ditemukan. Set environment variable ini!");
  Deno.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);
const model = genAI.getGenerativeModel({ model: "gemini-3-flash-preview" }); 

console.log("🚀 Server Chess AI (Gemini) berjalan di http://localhost:8000");

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);
  
  // CORS Headers
  const headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json"
  };

  if (request.method === "OPTIONS") {
    return new Response(null, { headers });
  }

  // GET / -> Health Check
  if (url.pathname === "/") {
    return new Response("Server Aktif 🚀", { 
        status: 200, 
        headers: { "Content-Type": "text/plain; charset=utf-8", "Access-Control-Allow-Origin": "*" } 
    });
  }

  if (url.pathname === "/analyze" && request.method === "POST") {
    try {
        const body = await request.json();
        const fen = body.fen;

        if (!fen) {
            return new Response(JSON.stringify({ error: "No FEN provided" }), { status: 400, headers });
        }

        console.log(`Received FEN: ${fen}`);

        // Prompt Engineering for Chess
        const prompt = `
        You are a Grandmaster Chess Engine.
        Current Board (FEN): ${fen}
        
        Task: Analyze the position and provide the absolute best move.
        Output Format: Just the move in UCI format (e.g., e2e4, g1f3) or standard notation if UCI is ambiguous, inside a JSON block.
        Do not explain.
        
        JSON:
        {"move": "..."}
        `;

        const result = await model.generateContent(prompt);
        const responseText = result.response.text();
        console.log("Gemini Raw:", responseText);

        // Extract JSON
        const jsonMatch = responseText.match(/\{.*"move".*\}/s);
        let move = "";
        if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            move = parsed.move;
        } else {
            // Fallback simplistic parsing
             move = responseText.trim();
        }

        return new Response(JSON.stringify({ move: move, fen: fen }), { headers });

    } catch (err) {
      console.error(err);
      return new Response(JSON.stringify({ error: (err as Error).message }), { status: 500, headers });
    }
  }

  return new Response("Not Found", { status: 404 });
}

Deno.serve({ port: 8000 }, handleRequest);
