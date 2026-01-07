import { GoogleGenerativeAI } from "npm:@google/generative-ai@0.21.0";
import { Chess } from "npm:chess.js@1.0.0";

const apiKey = Deno.env.get("GEMINI_API_KEY");
if (!apiKey) {
  console.error("❌ GEMINI_API_KEY tidak ditemukan. Set environment variable ini!");
  Deno.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" }); 

console.log("🚀 Server Chess AI (Gemini + chess.js) berjalan di http://localhost:8000");

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
    return new Response("Server Aktif 🚀 (with chess.js validation)", { 
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

        // Initialize chess.js with FEN
        const chess = new Chess(fen);
        const legalMoves = chess.moves({ verbose: true });
        const legalMovesUCI = legalMoves.map((m: { from: string; to: string; promotion?: string }) => 
            m.from + m.to + (m.promotion || '')
        );
        
        console.log(`Legal moves: ${legalMovesUCI.join(', ')}`);
        
        if (legalMovesUCI.length === 0) {
            return new Response(JSON.stringify({ error: "No legal moves (game over?)", move: "" }), { headers });
        }

        // Prompt with legal moves hint
        const prompt = `
        You are a Grandmaster Chess Engine.
        Current Board (FEN): ${fen}
        
        IMPORTANT: You MUST choose from these legal moves ONLY:
        ${legalMovesUCI.join(', ')}
        
        Task: Pick the absolute best move from the list above.
        Output Format: Just the move in UCI format (e.g., e2e4) inside a JSON block.
        
        JSON:
        {"move": "..."}
        `;

        const result = await model.generateContent(prompt);
        const responseText = result.response.text();
        console.log("Gemini Raw:", responseText);

        // Extract move from JSON
        const jsonMatch = responseText.match(/\{.*"move".*\}/s);
        let move = "";
        if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            move = parsed.move?.toLowerCase().replace(/[^a-h1-8qrbn]/g, '') || "";
        }

        // Validate move
        if (!legalMovesUCI.includes(move)) {
            console.log(`⚠️ Gemini gave illegal move: ${move}, picking random legal move`);
            move = legalMovesUCI[Math.floor(Math.random() * legalMovesUCI.length)];
        }

        console.log(`✅ Final move: ${move}`);
        return new Response(JSON.stringify({ move: move, fen: fen }), { headers });

    } catch (err) {
      console.error(err);
      return new Response(JSON.stringify({ error: (err as Error).message }), { status: 500, headers });
    }
  }

  return new Response("Not Found", { status: 404 });
}

Deno.serve({ port: 8000 }, handleRequest);
