import Link from 'next/link';
import NewsCard from './components/NewsCard';
import RaceCountdown from './components/RaceCountdown';

export default function Home() {
  const featuredNews = [
    {
      id: 1,
      title: "F1 2024 Season Preview",
      description: "Get ready for the new season with our comprehensive preview.",
      image: "/images/news1.jpg",
    },
    {
      id: 2,
      title: "New Regulations for 2024",
      description: "Learn about the new regulations that will shape the upcoming season.",
      image: "/images/news2.jpg",
    },
    {
      id: 3,
      title: "Driver Transfers and Rumors",
      description: "Stay updated on the latest driver transfers and rumors in F1.",
      image: "/images/news3.jpg",
    },
  ];
  
  return (
    <div className="space-y-10">
      <section className="relative h-[50vh] bg-gradient-to-r from-red-700 to-red-900 rounded-lg overflow-hidden">
        <div className="absolute inset-0 bg-[url('/images/f1-hero.jpg')] bg-cover bg-center opacity-30"></div>
        <div className="relative h-full flex flex-col justify-center items-center text-white text-center p-4">
          <h1 className="text-4xl md:text-6xl font-bold mb-4">Formula 1 Hub</h1>
          <p className="text-xl md:text-2xl max-w-2xl">Your ultimate destination for F1 news, schedules, and standings</p>
        </div>
      </section>
      
      <section>
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-3xl font-bold">Latest News</h2>
          <Link href="/news" className="text-red-600 hover:text-red-800">
            View all news →
          </Link>
        </div>
        <div className="grid md:grid-cols-3 gap-6">
          {featuredNews.map((news) => (
            <NewsCard key={news.id} news={news} />
          ))}
        </div>
      </section>
      
      <section className="bg-gray-100 p-6 rounded-lg">
        <h2 className="text-3xl font-bold mb-6">Next Race</h2>
        <RaceCountdown />
        <div className="mt-4">
          <Link href="/schedule" className="text-red-600 hover:text-red-800">
            View full schedule →
          </Link>
        </div>
      </section>
      
      <section>
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-3xl font-bold">Standings Highlights</h2>
          <Link href="/standings" className="text-red-600 hover:text-red-800">
            View complete standings →
          </Link>
        </div>
        {/* Standing highlights component would go here */}
      </section>
    </div>
  );
}