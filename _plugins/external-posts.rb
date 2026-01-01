require 'feedjira'
require 'httparty'
require 'jekyll'

module ExternalPosts
  class ExternalPostsGenerator < Jekyll::Generator
    safe true
    priority :high

    def generate(site)
      if site.config['external_sources'] != nil
        site.config['external_sources'].each do |src|
          begin
            p "Fetching external posts from #{src['name']}:"
            response = HTTParty.get(src['rss_url'], timeout: 10)

            unless response.success?
              p "...skipping (HTTP #{response.code})"
              next
            end

            xml = response.body
            feed = Feedjira.parse(xml)

            if feed.nil? || !feed.respond_to?(:entries)
              p "...skipping (invalid feed format)"
              next
            end

            feed.entries.each do |e|
              p "...fetching #{e.url}"
              slug = e.title.downcase.strip.gsub(' ', '-').gsub(/[^\w-]/, '')
              path = site.in_source_dir("_posts/#{slug}.md")
              doc = Jekyll::Document.new(
                path, { :site => site, :collection => site.collections['posts'] }
              )
              doc.data['external_source'] = src['name'];
              doc.data['feed_content'] = e.content;
              doc.data['title'] = "#{e.title}";
              doc.data['description'] = e.summary;
              doc.data['date'] = e.published;
              doc.data['redirect'] = e.url;
              site.collections['posts'].docs << doc
            end
          rescue StandardError => e
            p "...error fetching from #{src['name']}: #{e.message}"
            p "...continuing build without external posts"
          end
        end
      end
    end
  end

end
