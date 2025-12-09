"use client";
import { useEffect, useState } from "react";
import { precomputedTiles } from "@/utils/tilePattern";

type GitHubUser = { username: string; avatar: string };

const githubLinks = [
  "https://github.com/BrocodeADI",
  "https://github.com/teerthsharma",
  "https://github.com/farhadastro",
  "https://github.com/Debyte404",
];

export default function ContactPage() {
  const [profiles, setProfiles] = useState<GitHubUser[]>([]);

  useEffect(() => {
    (async () => {
      const data = await Promise.all(
        githubLinks.map(async (link) => {
          const username = link.split("github.com/")[1].replace("/", "");
          const res = await fetch(`https://api.github.com/users/${username}`);
          const json = await res.json();
          return { username: json.login, avatar: json.avatar_url };
        })
      );
      setProfiles(data);
    })();
  }, []);

  return (
    <main className="relative min-h-screen w-full flex flex-col items-center justify-center bg-[#0a0f0a] text-white overflow-hidden">
      {/* Conway-style green grid */}
      <div className="absolute inset-0 grid grid-cols-20 grid-rows-20 opacity-20 z-0">
        {precomputedTiles.map((isGreen, i) => (
          <div
            key={i}
            className={`w-full h-full ${
              isGreen ? "bg-green-600" : "bg-transparent"
            }`}
          />
        ))}
      </div>

      <div className="relative z-10 p-8 text-center">
        <h1 className="text-4xl font-bold mb-8 text-green-400">Contact Us</h1>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-10">
          {profiles.map((user) => (
            <a
              key={user.username}
              href={`https://github.com/${user.username}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col items-center p-6 bg-[#101510]/60 backdrop-blur-md rounded-xl border border-green-900 
                         shadow-lg hover:shadow-green-500/20 transition-all duration-300 transform hover:-translate-y-1 
                         hover:bg-[#101510]/80 cursor-pointer"
            >
              <img
                src={user.avatar}
                alt={user.username}
                className="w-28 h-28 rounded-full mb-4 border-4 border-green-600 shadow-lg hover:scale-105 transition-transform"
              />
              <p className="text-lg font-semibold text-green-400 hover:text-green-300">
                {user.username}
              </p>
            </a>
          ))}
        </div>
      </div>
    </main>
  );
}
