"use client";

import Image from "next/image";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import CircularText from "@/components/CircularText";

export default function AboutPage() {
  return (
    <main className="relative min-h-screen w-full bg-gradient-to-b from-[#0a0a0f] to-[#111112] text-gray-200 font-mono flex flex-col items-center px-4 py-16 overflow-hidden">
      {/* Subtle glow */}
      <div className="absolute inset-0 bg-gradient-radial from-purple-900/10 via-transparent to-transparent blur-3xl pointer-events-none" />

      {/* ASCII Header */}
      <pre className="text-purple-400 text-xs sm:text-sm mb-6 text-center leading-5 select-none">
{String.raw`
 _______  ______   _______          _________            _______ 
(  ___  )(  ___ \ (  ___  )|\     /|\__   __/  |\     /|(  ____ \
| (   ) || (   ) )| (   ) || )   ( |   ) (     | )   ( || (    \/
| (___) || (__/ / | |   | || |   | |   | |     | |   | || (_____ 
|  ___  ||  __ (  | |   | || |   | |   | |     | |   | |(_____  )
| (   ) || (  \ \ | |   | || |   | |   | |     | |   | |      ) |
| )   ( || )___) )| (___) || (___) |   | |     | (___) |/\____) |
|/     \||/ \___/ (_______)(_______)   )_(     (_______)\_______)
                                                                 
`}
      </pre>

      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-3xl sm:text-5xl font-bold text-center text-purple-400 mb-6"
      >
        Debenture — Built by Team Esoteric
      </motion.h1>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="max-w-3xl text-center text-gray-400 mb-10 leading-relaxed"
      >
        <span className="text-purple-400 font-semibold">Debenture</span> was created during the 
        <span className="text-purple-300"> MUJ Hack X</span> hackathon — a 48-hour sprint where we re-imagined
        how people interact with financial systems. It’s an AI-driven loan sales assistant that brings
        transparency, empathy, and intelligence into digital finance using Groq LLM, FastAPI, and ChromaDB.
      </motion.p>

      {/* Problem Statement Screenshot
      <Card className="bg-[#151515]/70 border border-purple-700/30 w-full max-w-4xl mb-12">
        <CardContent className="p-4 sm:p-6 flex flex-col items-center gap-3">
          <h2 className="text-xl sm:text-2xl font-semibold text-purple-400 mb-2">
            🧩 Problem Statement
          </h2>
          <Image
            src="/assets/problem-statement.png"
            alt="Problem Statement Screenshot"
            width={900}
            height={500}
            className="rounded-lg border border-purple-600/40 shadow-lg object-contain"
          />
          <p className="text-sm text-gray-400 text-center mt-3">
            The challenge: simplify personal-loan journeys with ethical, conversational AI that adapts
            to each user’s context in real time.
          </p>
        </CardContent>
      </Card> */}
      <div className="my-12">
        <CircularText
            text="dream*debug*deliver*"
            onHover="slowDown"
            spinDuration={20}
            className="mb-3"
            />
        </div>
      {/* Team Section */}
      <div className="w-full max-w-5xl text-center">
        <h2 className="text-2xl sm:text-3xl font-semibold text-purple-400 mb-4">👨‍💻 Team Esoteric</h2>
        <p className="text-gray-400 mb-6">
          A collective of hackers, designers, and builders who turned midnight caffeine into a fully
          functional AI product.
        </p>

        <div className="w-full flex justify-center mb-8">
          <Image
            src="/assets/group.jpeg"
            alt="Team Esoteric Group Photo"
            width={900}
            height={500}
            className="rounded-2xl border border-purple-700/40 shadow-xl object-cover"
          />
        </div>
      </div>

      {/* Build Stack Section */}
      <Card className="bg-[#151515]/70 border border-purple-700/30 w-full max-w-4xl">
        <CardContent className="p-6">
          <h2 className="text-2xl sm:text-3xl font-semibold text-purple-400 mb-4 text-center">
            🛠️ Tech Stack & Tools
          </h2>
          <ul className="grid sm:grid-cols-2 gap-2 text-gray-300 text-sm sm:text-base">
            <li>⚡ FastAPI – Backend Framework</li>
            <li>🧠 Groq LLM – Conversational Engine</li>
            <li>🗄️ ChromaDB – Vector Memory & RAG</li>
            <li>🔌 WebSockets – Real-time Chat Link</li>
            <li>🎨 Next.js + ShadCN UI – Frontend</li>
            <li>🐳 Docker – Containerized Deployment</li>
          </ul>
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="mt-12 text-center text-xs text-gray-600">
        <p>
          Crafted with 💜 & caffeine during MUJ Hack X | © 2025 Team Esoteric
        </p>
      </div>
    </main>
  );
}
